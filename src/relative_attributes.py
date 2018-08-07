import tensorflow as tf
import numpy as np
import Dataset
import argparse
from GNNmodel import GNN_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='pubfig')
parser.add_argument('--attr', type=int, default=0)
args = parser.parse_args()

img_number = 5
batch_size = 10
max_epoch = 1000
learning_rate = 0.0001  # 0.0001
img_size = 227

def run_test(sess, dataset, epoch, img_list, label_list, test_accuracy):
	if args.dataset_name == 'pubfig':
		test_batch = dataset.test_batch(34)
	elif args.dataset_name == 'osr':
		test_batch = dataset.test_batch(137)
	accuracy = sess.run(test_accuracy, feed_dict={img_list[0]: test_batch[0][0], img_list[1]: test_batch[0][1], \
		                          img_list[2]: test_batch[0][2], img_list[3]: test_batch[0][3],
		                          img_list[4]: test_batch[0][4], \
		                          label_list[0]: test_batch[1][0], label_list[1]: test_batch[1][1],
		                          label_list[2]: test_batch[1][2], \
		                          label_list[4]: test_batch[1][4], label_list[3]: test_batch[1][3]})
	print "epoch: %d, test_accuracy:%f " %(epoch, accuracy)


def run_train(sess, train_step, img_list, label_list, datasetname, attributes_num, loss, test_accuracy):
	dataset = Dataset.reader(datasetname, attributes_num)
	epoch = 0
	step = 0
	run_test(sess, dataset, epoch, img_list, label_list, test_accuracy)
	while (epoch <= max_epoch):
		next_batch, epoch_end = dataset.next_batch(batch_size)
		batch_dict = {img_list[0]: next_batch[0][0], img_list[1]: next_batch[0][1], \
		                          img_list[2]: next_batch[0][2], img_list[3]: next_batch[0][3],
		                          img_list[4]: next_batch[0][4], \
		                          label_list[0]: next_batch[1][0], label_list[1]: next_batch[1][1],
		                          label_list[2]: next_batch[1][2], \
		                          label_list[4]: next_batch[1][4], label_list[3]: next_batch[1][3]}

		train_step.run(feed_dict= batch_dict)

		if step % 10 == 0:
			print "epoch: %d, step: %d, loss: %f, train_accuracy:%f" %(epoch, step, sess.run(loss, feed_dict=batch_dict), \
			                                                           sess.run(test_accuracy, feed_dict=batch_dict))

		step += 1
		if epoch_end == 1:
			run_test(sess, dataset, epoch, img_list, label_list, test_accuracy)
			epoch = epoch + 1
			step = 0


def run_GNN():
	model = GNN_model()

	# placeholder
	img_0 = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
	img_1 = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
	img_2 = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
	img_3 = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
	img_4 = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
	label_0 = tf.placeholder(tf.float32, shape=[None])
	label_1 = tf.placeholder(tf.float32, shape=[None])
	label_2 = tf.placeholder(tf.float32, shape=[None])
	label_3 = tf.placeholder(tf.float32, shape=[None])
	label_4 = tf.placeholder(tf.float32, shape=[None])

	img_list = [img_0, img_1, img_2, img_3, img_4]
	label_list = [label_0, label_1, label_2, label_3, label_4]

	# get inference model
	final_output_list = model.inference(img_list)

	# calculate loss
	loss = 0
	count_correct = 0
	count_wrong = 0
	for i in range(len(final_output_list)):
		for j in range(len(final_output_list)):
			if i < j:
				P = tf.clip_by_value(tf.sigmoid(final_output_list[i] - final_output_list[j]), 0.00001, 0.99999)
				L = tf.clip_by_value(label_list[i] - label_list[j], -1, 1) / 2.0 + 0.5
				loss = loss + tf.reduce_sum(- L * tf.log(P) - (1 - L) * tf.log(1 - P))
				count_correct = count_correct + tf.reduce_sum(tf.to_float(
					tf.greater((final_output_list[i] - final_output_list[j]) * (label_list[i] - label_list[j]), 0)))
				count_wrong = count_wrong + tf.reduce_sum(tf.to_float(
					tf.less((final_output_list[i] - final_output_list[j]) * (label_list[i] - label_list[j]), 0)))
	test_accuracy = count_correct / (count_correct + count_wrong)

	reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
	reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
	loss = loss + reg_term

	# train step and configuration
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True

	# run session
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		run_train(sess, train_step, img_list, label_list, args.dataset_name, args.attr, loss, test_accuracy)


if __name__ == '__main__':
	run_GNN()
