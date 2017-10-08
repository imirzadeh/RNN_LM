import sys
import math
import numpy as np
import tensorflow as tf

class LanguageModel(object):
	def __init__(self):
		self.hidden_size = 50
		self.cell_layers = 3
		self.dropout = 0.5

	def read_vectors(self, file_name="vectors.txt"):
		all_vectors = []
		mappings = {}
		counter = 2
		mappings['UNK'] = 0
		mappings['DOT'] = 1
		all_vectors.append(None)
		all_vectors.append(None)

		with open(file_name, 'r') as f:
			for line in f.readlines():
				try:
					vector = line.split(' ')
					mappings[vector[0]] = counter
					vector = [float(x) for i, x in enumerate(vector) if i != 0]
					all_vectors.append(vector)
					counter += 1
				except Exception as e:
					print(e)
					print("Error parsing line", line)
		self.vocab_size = len(all_vectors)
		self.vector_size = len(all_vectors[2])
		all_vectors[0] = [0.0 for i in range(self.vector_size)]
		all_vectors[1] = [1.0 for i in range(self.vector_size)]
		self.all_vectors = all_vectors
		self.mappings = mappings

	def get_token_index(self, token):
		if token == '.':
			return 1
		return self.mappings.get(token, 0)

	def _read_data(self, file_name):
		inputs = []
		labels = []
		lengths = []
		max_length = 0
		print("reading {0}".format(file_name))
		with open(file_name, 'r') as f:
			for line in f.readlines():
				l = line.split(' ')
				l = [self.get_token_index(token) for token in l]
				if len(l) == 0:
					continue
				l.append(self.get_token_index('DOT'))
				inputs.append(l[:-1])
				labels.append(l[1:])
				_len = len(l) - 1
				max_length = _len if _len > max_length else max_length
				lengths.append(_len)
		return inputs, labels, lengths, max_length

	def _pad(self, arr, l):
		return [x + [1] * (l-len(x)) for x in arr]

	def read_train_data(self, file_name="train.txt"):
		(inputs, labels, lengths, m) = self._read_data(file_name)
		self.train_inputs  = np.array(self._pad(inputs, m))
		self.train_labels  = np.array(self._pad(labels, m))
		self.train_lengths = np.array(lengths)


	def read_validation_data(self, file_name="validation.txt"):
		(inputs, labels, lengths, m) = self._read_data(file_name)
		self.validation_inputs  = np.array(self._pad(inputs, m))
		self.validation_labels  = np.array(self._pad(labels, m))
		self.validation_lengths = np.array(lengths)

	def _init_embeddings(self):
		self.embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.vector_size]), trainable=False, name="embedding")
		self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.vector_size])
		self.embedding_assign_op = self.embedding.assign(self.embedding_placeholder)


	def _init_cell(self):
		cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
		cell = tf.nn.rnn_cell.MultiRNNCell( [cell] * self.cell_layers )

		self.cell = cell


	def _init_vars(self):
		X = tf.placeholder(tf.int32, [None, None]) # (batch, seq)
		Y = tf.placeholder(tf.int32, [None, None]) # (tch, seq)
		L = tf.placeholder(tf.int32, [None]) # (batch)

		inputs = tf.nn.embedding_lookup(self.embedding, X) # (batch, seq, vec_size)

		with tf.variable_scope('softmax'):
			W_softmax = tf.get_variable('W_softmax', [self.hidden_size, self.vocab_size])
			b_softmax = tf.get_variable('b_softmax', [self.vocab_size])

		output, state = tf.nn.dynamic_rnn(self.cell, inputs, dtype=tf.float32, sequence_length=L)
		#output (batch, size, hidden_size)

		## F**K YOU TENSORFLOW! BROADCAST MATMUL!!
		output_ = tf.reshape(output, [-1, self.hidden_size]) # (batch * seq, hidden)
		result_ = tf.matmul(output_, W_softmax) + b_softmax # (batch * seq, vocab)
		output_shape = tf.gather(tf.shape(output), [0, 1])
		target_shape = tf.concat(0, [output_shape, [self.vocab_size]]) # =[batch, seq, vocab]
		result = tf.reshape(result_, target_shape) # (batch, seq, vocab)

		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=Y)
		loss = self.loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))

		tf.summary.scalar('cross_entropy', loss)

		pred_labels = tf.cast(tf.argmax(result, axis=2), tf.int32) # (b,s)
		correct_pred = tf.cast(tf.equal(Y, pred_labels), tf.float32) # (b, s)
		self.accuracy = tf.reduce_mean(correct_pred)

		tf.summary.scalar('accuracy', self.accuracy)

		# AdamOptimizer(self.learning_rate) ?
		self.train_op = tf.train.AdamOptimizer().minimize(loss)

		self.X = X
		self.Y = Y
		self.L = L

	def init(self):
		self._init_embeddings()
		self._init_cell()
		self._init_vars()

	def _feed(self, inputs, outputs, lengths):
		return {self.X: inputs, self.Y: outputs, self.L: lengths}

	def train(self, batch_size = 16, epoch = 30, board_addr='./tensor_board_logs', checkpoint_count=10):
		batch_count = math.floor(len(self.train_inputs) / batch_size)
		print("batch count {0}".format(batch_count))

		with tf.Session() as sess:
			writer = tf.summary.FileWriter(board_addr, sess.graph)
			saver = tf.train.Saver()


			sess.run(tf.global_variables_initializer())
			sess.run(self.embedding_assign_op, feed_dict={self.embedding_placeholder: self.all_vectors})

			merged = tf.summary.merge_all()

			for e in range(epoch):
				print("epoch {0}".format(e))
				for batch in range(batch_count):
					start = batch * batch_size
					end   = (batch + 1) * batch_size
					X_feed =  self.train_inputs[start:end]
					Y_feed =  self.train_labels[start:end]
					L_feed = self.train_lengths[start:end]

					_, loss = sess.run([self.train_op, self.loss], feed_dict=self._feed(X_feed, Y_feed, L_feed))
					sys.stdout.write("  loss: %f [batch:%d]\r" % (loss, batch))
					sys.stdout.flush()

				print()
				value, summary = sess.run([self.accuracy, merged], feed_dict=self._feed(self.validation_inputs, self.validation_labels, self.validation_lengths))
				writer.add_summary(summary, e)
				print(">> accuracy {0}".format(value))
				if e == epoch - 1 or e % checkpoint_count == 0:
					saver.save(sess, board_addr + '/model.ckpt', e)

		writer.close()

if __name__ == '__main__':
	lm = LanguageModel()
	lm.read_vectors()
	lm.read_train_data()
	lm.read_validation_data()
	lm.init()
	lm.train(epoch=20)
