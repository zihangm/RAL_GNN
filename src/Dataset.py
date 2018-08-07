import numpy as np
import os
import scipy.misc
import random
import re

OSR_matrix = [
	{'tallbuilding': 1, 'insidecity': 2, 'street': 2, 'highway': 3, 'coast': 4, 'opencountry': 4, 'mountain': 4,
	 'forest': 4}]
OSR_matrix.append(
	{'tallbuilding': 1, 'insidecity': 2, 'street': 2, 'highway': 4, 'coast': 4, 'opencountry': 4, 'mountain': 3,
	 'forest': 1})
OSR_matrix.append(
	{'tallbuilding': 7, 'insidecity': 5, 'street': 6, 'highway': 4, 'coast': 2, 'opencountry': 1, 'mountain': 3,
	 'forest': 3})
OSR_matrix.append(
	{'tallbuilding': 5, 'insidecity': 3, 'street': 3, 'highway': 4, 'coast': 4, 'opencountry': 2, 'mountain': 2,
	 'forest': 1})
OSR_matrix.append(
	{'tallbuilding': 6, 'insidecity': 4, 'street': 4, 'highway': 5, 'coast': 3, 'opencountry': 2, 'mountain': 2,
	 'forest': 1})
OSR_matrix.append(
	{'tallbuilding': 4, 'insidecity': 4, 'street': 4, 'highway': 4, 'coast': 1, 'opencountry': 3, 'mountain': 2,
	 'forest': 4})

pubfig_matrix = []
pubfig_matrix.append(
	{'AlexRodriguez': 6, 'CliveOwen': 8, 'HughLaurie': 7, 'JaredLeto': 5, 'MileyCyrus': 2, 'ScarlettJohansson': 1,
	 'ViggoMortensen': 4, 'ZacEfron': 3})
pubfig_matrix.append(
	{'AlexRodriguez': 1, 'CliveOwen': 2, 'HughLaurie': 3, 'JaredLeto': 5, 'MileyCyrus': 7, 'ScarlettJohansson': 6,
	 'ViggoMortensen': 8, 'ZacEfron': 4})
pubfig_matrix.append(
	{'AlexRodriguez': 5, 'CliveOwen': 3, 'HughLaurie': 2, 'JaredLeto': 4, 'MileyCyrus': 8, 'ScarlettJohansson': 6,
	 'ViggoMortensen': 1, 'ZacEfron': 7})
pubfig_matrix.append(
	{'AlexRodriguez': 4, 'CliveOwen': 4, 'HughLaurie': 3, 'JaredLeto': 1, 'MileyCyrus': 6, 'ScarlettJohansson': 5,
	 'ViggoMortensen': 2, 'ZacEfron': 5})
pubfig_matrix.append(
	{'AlexRodriguez': 8, 'CliveOwen': 4, 'HughLaurie': 3, 'JaredLeto': 2, 'MileyCyrus': 6, 'ScarlettJohansson': 7,
	 'ViggoMortensen': 1, 'ZacEfron': 5})
pubfig_matrix.append(
	{'AlexRodriguez': 5, 'CliveOwen': 5, 'HughLaurie': 5, 'JaredLeto': 1, 'MileyCyrus': 3, 'ScarlettJohansson': 4,
	 'ViggoMortensen': 5, 'ZacEfron': 2})
pubfig_matrix.append(
	{'AlexRodriguez': 6, 'CliveOwen': 7, 'HughLaurie': 5, 'JaredLeto': 8, 'MileyCyrus': 1, 'ScarlettJohansson': 2,
	 'ViggoMortensen': 4, 'ZacEfron': 3})
pubfig_matrix.append(
	{'AlexRodriguez': 4, 'CliveOwen': 6, 'HughLaurie': 5, 'JaredLeto': 2, 'MileyCyrus': 1, 'ScarlettJohansson': 3,
	 'ViggoMortensen': 7, 'ZacEfron': 8})
pubfig_matrix.append(
	{'AlexRodriguez': 1, 'CliveOwen': 2, 'HughLaurie': 8, 'JaredLeto': 3, 'MileyCyrus': 3, 'ScarlettJohansson': 4,
	 'ViggoMortensen': 3, 'ZacEfron': 7})
pubfig_matrix.append(
	{'AlexRodriguez': 7, 'CliveOwen': 5, 'HughLaurie': 1, 'JaredLeto': 2, 'MileyCyrus': 6, 'ScarlettJohansson': 8,
	 'ViggoMortensen': 3, 'ZacEfron': 4})
pubfig_matrix.append(
	{'AlexRodriguez': 6, 'CliveOwen': 4, 'HughLaurie': 1, 'JaredLeto': 3, 'MileyCyrus': 8, 'ScarlettJohansson': 7,
	 'ViggoMortensen': 2, 'ZacEfron': 5})


OSR_train_test_split = 2000
pubfig_train_test_split = 600


class reader:
	def __init__(self, datasetname, attributes_num):
		self.dataset = []
		self.trainset = []
		self.testset = []
		if datasetname == 'osr':
			self.name = 'osr'
			self.imgdir = './dataset/osr/'
		elif datasetname == 'pubfig':
			self.name = 'pubfig'
			self.imgdir = './dataset/pubfig/'
		self.attribute_number = attributes_num
		self.batch_pointer = 0
		self.test_pointer = 0
		self.global_mean = self.get_global_mean()
		self.read_img()

	def get_global_mean(self):
		img_name_list = os.listdir(self.imgdir)
		img_list = []
		if self.name == 'osr':
			global_ave_num = OSR_train_test_split
		elif self.name == 'pubfig':
			global_ave_num = pubfig_train_test_split
		for i in range(global_ave_num):  # for now, just use mean of first 1000
			img = scipy.misc.imresize(scipy.misc.imread(self.imgdir + img_name_list[i]), [256, 256])
			img_list.append(img)
		global_mean = np.mean(np.array(img_list))
		return global_mean

	def rand_crop(self, img):
		flip = random.randint(0, 1)
		if flip == 1:
			img[:, :, 0] = np.fliplr(img[:, :, 0])
			img[:, :, 1] = np.fliplr(img[:, :, 1])
			img[:, :, 2] = np.fliplr(img[:, :, 2])
		rand_x = random.randint(0, 23)
		rand_y = random.randint(0, 23)
		return img[rand_x:rand_x + 227, rand_y:rand_y + 227, :]

	def crop(self, img, k):
		if k >= 5:
			k = k - 5
			img[:, :, 0] = np.fliplr(img[:, :, 0])
			img[:, :, 1] = np.fliplr(img[:, :, 1])
			img[:, :, 2] = np.fliplr(img[:, :, 2])
		if k == 0:
			return img[8:235, 8:235, :]
		if k == 1:
			return img[0:0 + 227, 0:0 + 227, :]
		if k == 2:
			return img[0:0 + 227, 23:23 + 227, :]
		if k == 3:
			return img[23:23 + 227, 0:0 + 227, :]
		if k == 4:
			return img[23:23 + 227, 23:23 + 227, :]

	def read_img(self):
		for img_name in os.listdir(self.imgdir):
			img = scipy.misc.imresize(scipy.misc.imread(self.imgdir + img_name), [256, 256])
			img = img - np.float32(self.global_mean)
			img[:, :, 2], img[:, :, 0] = img[:, :, 0], img[:, :, 2]
			img_tag = img_name.split('_')[0]
			self.dataset.append([img, img_tag])
		random.shuffle(self.dataset)
		random.shuffle(self.dataset)
		if self.name == 'osr':
			trainsize = OSR_train_test_split
		elif self.name == 'pubfig':
			trainsize = pubfig_train_test_split
		self.trainset = self.dataset[0:trainsize]
		self.testset = self.dataset[trainsize:]

	def next_batch(self, batch_size):
		next_batch = [[], []]
		epoch_end = 0
		for i in range(5):
			data_batch = self.trainset[self.batch_pointer:(self.batch_pointer + batch_size)]
			self.batch_pointer = self.batch_pointer + batch_size
			img_batch = [self.rand_crop(data_batch[j][0]) for j in range(batch_size)]
			label_batch = self.get_label(data_batch)
			next_batch[0].append(np.array(img_batch))
			next_batch[1].append(np.array(label_batch))
			if self.name == 'osr':
				trainsize = OSR_train_test_split
			elif self.name == 'pubfig':
				trainsize = pubfig_train_test_split
			if self.batch_pointer >= trainsize - batch_size:
				self.batch_pointer = 0
				epoch_end = 1
				random.shuffle(self.trainset)
		return next_batch, epoch_end

	def get_label(self, batch_data):
		if self.name == 'osr':
			batch_label = [OSR_matrix[self.attribute_number][batch_data[i][1]] for i in range(len(batch_data))]
		elif self.name == 'pubfig':
			batch_label = [pubfig_matrix[self.attribute_number][batch_data[i][1]] for i in range(len(batch_data))]
		return batch_label

	def test_batch(self, batch_size):
		self.test_pointer = 0
		next_batch = [[], []]
		for i in range(5):
			data_batch = self.testset[self.test_pointer:(self.test_pointer + batch_size)]
			img_batch = [self.crop(data_batch[j][0], 0) for j in range(batch_size)]
			label_batch = self.get_label(data_batch)
			next_batch[0].append(np.array(img_batch))
			next_batch[1].append(np.array(label_batch))
			self.test_pointer = self.test_pointer + batch_size
		return next_batch
