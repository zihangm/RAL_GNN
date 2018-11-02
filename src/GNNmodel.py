
import tensorflow as tf
import numpy as np

#load pretrained weight
net_data = np.load("./pretrain/bvlc_alexnet.npy",encoding='latin1').item()

regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

def weight_variable(shape):  # use the xavier initializer
	return tf.get_variable('weight', shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
	                       regularizer=regularizer)

def bias_variable(shape):
	initial = tf.constant(0.01, shape=shape)
	return tf.get_variable('bias', initializer=initial, regularizer=regularizer)


def conv(inputt, kernel, biases, c_o, s_h, s_w, padding="VALID", group=1):
	c_i = inputt.get_shape()[-1]
	assert c_i % group == 0
	assert c_o % group == 0
	convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
	if group == 1:
		conv = convolve(inputt, kernel)
	else:
		input_groups = tf.split(inputt, group, 3)
		kernel_groups = tf.split(kernel, group, 3)
		output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
		conv = tf.concat(output_groups, 3)
	return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class GNN_model(object):
	def __init__(self):
		pass

	def inference(self, img_list):
		nodes_list = self.encoder(img_list)

		edges_list = self.get_edges(nodes_list)

		with tf.variable_scope('message_time_1') as message_1:
			message_list_1 = []
			count_message_1 = 0
			for i in range(len(nodes_list)):
				if (count_message_1 == 1):
					message_1.reuse_variables()
				message_list_1.append(self.Message_func(i, nodes_list, edges_list))
				count_message_1 = count_message_1 + 1

		with tf.variable_scope('gru_time_1') as gru_t1:
			gru_list_1 = []
			gru_count_1 = 0
			gru_cell_1 = tf.contrib.rnn.GRUCell(4096, activation=tf.nn.tanh)
			for i in range(len(nodes_list)):
				if (gru_count_1 == 1):
					gru_t1.reuse_variables()
				_, h_t1 = gru_cell_1(message_list_1[i],
				                     nodes_list[i])  # two outputs, "output" and "new_state"(although the same)
				gru_list_1.append(h_t1)
				gru_count_1 = gru_count_1 + 1

		nodes_out_list = gru_list_1
		final_output_list = self.read_out(nodes_out_list)
		return final_output_list

	def Single_message_func(self, h_vt, h_wt, e_vw, edge_dir):
		if edge_dir == 1:
			with tf.variable_scope('scope4'):
				cat = tf.concat([h_wt, e_vw], 1)
				W_fc1 = weight_variable([4096 + 4096, 4096])
				b_fc1 = bias_variable([4096])
				fc1 = tf.nn.relu(tf.matmul(cat, W_fc1) + b_fc1)
				#fc1_out = tf.nn.dropout(fc1, keep_prob)
		return fc1

	# Message function to compute the message for one node
	def Message_func(self, v, n_list, e_list):
		with tf.variable_scope('scope3') as vs3:
			tmp_list = []
			count = 0  # to represent this is the ?th time calling the Single func
			for i in range(len(n_list)):
				if i != v:
					if count == 1:
						vs3.reuse_variables()
					tmp_list.append(self.Single_message_func(n_list[v], n_list[i], e_list[i][v], 1))
					#take it as undirected graph
					count = count + 1
			m_sum = 0
			for i in range(len(tmp_list)):
				m_sum = m_sum + tmp_list[i]
			return m_sum  # m_sum is the total message for a single node

	def get_edges(self, n_list):
		with tf.variable_scope('edges') as edges:
			edges_list = [[] for i in range(len(n_list))]
			count = 0
			for i in range(len(n_list)):
				for j in range(len(n_list)):
					if count == 1:
						edges.reuse_variables()
					count = count + 1
					W_edge = weight_variable([4096, 4096])
					b_edge = bias_variable([4096])
					h_edge = tf.nn.relu(tf.matmul(tf.abs(n_list[i] - n_list[j]),
					                              W_edge) + b_edge)  # To keep the edge representation symmetric
					#h_edge_out = tf.nn.dropout(h_edge, keep_prob)
					edges_list[i].append(h_edge)
			return edges_list

	def read_out(self, n_list):
		with tf.variable_scope('read_out') as read_out:
			out_list = []
			count = 0
			for i in range(len(n_list)):
				if count == 1:
					read_out.reuse_variables()
				W_out = weight_variable([4096, 1])
				b_out = bias_variable([1])
				h_out = tf.reshape(tf.matmul(n_list[i], W_out) + b_out, [-1])
				out_list.append(h_out)
				count = count + 1
			return out_list

	# the encoder of images, takes in a list of images
	def encoder(self, img_l):
		nodes_list = []
		with tf.variable_scope('encoder_img') as en_img:
			count = 0
			for i in range(len(img_l)):
				count = count + 1
				if count == 2:
					en_img.reuse_variables()
				with tf.variable_scope('img_layer1'):
					W_conv1 = tf.get_variable('weight', initializer=net_data["conv1"][0], regularizer=regularizer)
					b_conv1 = tf.get_variable('bias', initializer=net_data["conv1"][1], regularizer=regularizer)
					h_conv1 = tf.nn.relu(
						tf.nn.conv2d(img_l[i], W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1)
					radius = 2;
					alpha = 2e-05;
					beta = 0.75;
					bias = 1.0
					lrn1 = tf.nn.local_response_normalization(h_conv1, depth_radius=radius, alpha=alpha, beta=beta,
					                                          bias=bias)
					h_pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
				with tf.variable_scope('img_layer2'):
					c_o = 256;
					s_h = 1;
					s_w = 1;
					group = 2
					W_conv2 = tf.get_variable('weight', initializer=net_data["conv2"][0], regularizer=regularizer)
					b_conv2 = tf.get_variable('bias', initializer=net_data["conv2"][1], regularizer=regularizer)
					conv2_in = conv(h_pool1, W_conv2, b_conv2, c_o, s_h, s_w, padding="SAME", group=group)
					h_conv2 = tf.nn.relu(conv2_in)
					radius = 2;
					alpha = 2e-05;
					beta = 0.75;
					bias = 1.0
					lrn2 = tf.nn.local_response_normalization(h_conv2, depth_radius=radius, alpha=alpha, beta=beta,
					                                          bias=bias)
					h_pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
				with tf.variable_scope('img_layer3'):
					W_conv3 = tf.get_variable('weight', initializer=net_data["conv3"][0], regularizer=regularizer)
					b_conv3 = tf.get_variable('bias', initializer=net_data["conv3"][1], regularizer=regularizer)
					h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
				with tf.variable_scope('img_layer4'):
					c_o = 384;
					s_h = 1;
					s_w = 1;
					group = 2
					W_conv4 = tf.get_variable('weight', initializer=net_data["conv4"][0], regularizer=regularizer)
					b_conv4 = tf.get_variable('bias', initializer=net_data["conv4"][1], regularizer=regularizer)
					conv4_in = conv(h_conv3, W_conv4, b_conv4, c_o, s_h, s_w, padding="SAME", group=group)
					h_conv4 = tf.nn.relu(conv4_in)
				with tf.variable_scope('img_layer5'):
					c_o = 256;
					s_h = 1;
					s_w = 1;
					group = 2
					W_conv5 = tf.get_variable('weight', initializer=net_data["conv5"][0], regularizer=regularizer)
					b_conv5 = tf.get_variable('bias', initializer=net_data["conv5"][1], regularizer=regularizer)
					conv5_in = conv(h_conv4, W_conv5, b_conv5, c_o, s_h, s_w, padding="SAME", group=group)
					h_conv5 = tf.nn.relu(conv5_in)
					h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
				# fc1, to 4096
				with tf.variable_scope('img_fc1'):
					W_fc1 = tf.get_variable('weight', initializer=net_data["fc6"][0], regularizer=regularizer)
					b_fc1 = tf.get_variable('bias', initializer=net_data["fc6"][1], regularizer=regularizer)
					h_fc1 = tf.nn.relu_layer(tf.reshape(h_pool5, [-1, int(np.prod(h_pool5.get_shape()[1:]))]), W_fc1,
					                         b_fc1)
				with tf.variable_scope('img_fc2'):
					W_fc2 = tf.get_variable('weight', initializer=net_data["fc7"][0], regularizer=regularizer)
					b_fc2 = tf.get_variable('bias', initializer=net_data["fc7"][1], regularizer=regularizer)
					h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
				#append the nodes_list
				nodes_list.append(h_fc2)
		#     tf.summary.histogram('b_conv6', b_conv6)
		return nodes_list
