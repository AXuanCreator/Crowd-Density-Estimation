import os
import glob
import random

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.measure import block_reduce

from utils import generate_h5
from config import cfg


class CustomDataset(Dataset):
	def __init__(self, root, mode, net, scaling=None, transform=None):
		self.root = root
		self.mode = mode
		self.net = net
		self.image_path = os.path.join(self.root, 'images')
		self.gt_path = os.path.join(self.root, 'ground_truth')
		self.image_list = sorted(glob.glob(self.image_path + '/*.jpg'))
		self.gt_list = None
		self.transform = transform
		self.scaling = scaling
		self._prepare_gt()

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		image = cv2.imread(self.image_list[idx])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		gt = np.asarray(h5py.File(self.gt_list[idx], 'r')['density'])
		rig = None

		if self.net == 'p2p':
			image, gt = self._data_downsample(image, gt, downsample=4)

		# rgb&gt图输出
		if cfg.DEBUG:
			combine_image = np.hstack((image, np.repeat(np.expand_dims(gt, axis=-1), 3, axis=-1) * 255))
			cv2.imwrite(f'./debug/image&gt_{cfg.CURRENT_TIME}.png', combine_image)

		# 数据增强
		if self.mode == 'train':
			image, gt, rig = self._data_enhancement(image, gt)

		rgb = image.copy()
		info = f"{self.mode}_{os.path.basename(self.image_list[idx]).split('_')[1].split('.')[0]}_{rig}"

		image = self.transform(image.copy()) if self.transform else transforms.ToTensor()(image.copy()) / 255

		if self.net in ['can', 'can-alex']:
			batch = {'image': image, 'gt': gt, 'info': info, 'rgb': rgb}
		elif self.net == 'p2p':
			labels = gt.flatten()
			points = np.indices(gt.shape).reshape(2, -1).T
			batch = {'image': image, 'gt': {'labels': labels, 'points': points}, 'info': info, 'rgb': rgb}
		else:
			assert self.net is None, 'Net provision required'

		return batch

	def _prepare_gt(self):
		if self.net in ['can', 'can-alex']:
			if len(glob.glob(self.gt_path + '/*can*.h5')) != len(self.image_list):
				print('Generating .h5 from .mat for can/can-alex')
				generate_h5(self.gt_path, *cv2.imread(self.image_list[0]).shape[:2], net='can')
			self.gt_list = sorted(glob.glob(self.gt_path + '/*can*.h5'))
		elif self.net == 'p2p':
			if len(glob.glob(self.gt_path + '/*p2p*.h5')) != len(self.image_list):
				print('Generating .h5 from .mat for p2p')
				generate_h5(self.gt_path, *cv2.imread(self.image_list[0]).shape[:2], net='p2p')
			self.gt_list = sorted(glob.glob(self.gt_path + '/*p2p*.h5'))
		else:
			assert self.net is None, 'Net provision required'

	def _data_enhancement(self, image, gt):
		"""
		数据增强
		"""
		ratio = 0.5
		h, w = image.shape[:2]
		rdn = random.random()
		rig = ['0_0', '1_0', '0_1', '1_1'][int(rdn * 4)]
		dx = int(w * ratio * (rig[0] == '1'))
		dy = int(h * ratio * (rig[2] == '1'))
		crop_x = int(dx + w * ratio)
		crop_y = int(dy + h * ratio)
		crop_image = image[dy:crop_y, dx:crop_x, :]
		crop_gt = gt[dy:crop_y, dx:crop_x]

		if random.random() > 0.8:
			rig += '_fliplr'
			crop_image = np.fliplr(crop_image)
			crop_gt = np.fliplr(crop_gt)

		if self.scaling is not None and self.net != 'p2p':
			crop_gt = cv2.resize(crop_gt, (
			int(crop_gt.shape[1] / 8), int(crop_gt.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * self.scaling ** 2

		return crop_image, crop_gt, rig

	def _data_downsample(self, image, gt, downsample=4):
		image = cv2.resize(image, (
		image.shape[1] // downsample, image.shape[0] // downsample), interpolation=cv2.INTER_CUBIC)
		gt = block_reduce(gt, block_size=(downsample, downsample), func=np.max)
		return image, gt
