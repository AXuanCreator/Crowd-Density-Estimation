import os
import glob
import random

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from utils import generate_h5
from config import cfg


class CustomDataset(Dataset):
	def __init__(self, root, mode, scaling=None, transform=None):
		self.root = root
		self.image_path = os.path.join(self.root, 'images')
		self.gt_path = os.path.join(self.root, 'ground_truth')

		self.image_list = sorted(glob.glob(self.image_path + '/*.jpg'))
		self.gt_list = sorted(glob.glob(self.gt_path + '/*.h5'))

		if len(self.gt_list) == 0:
			print('generate .h5 from .mat')
			generate_h5(self.gt_path, *cv2.imread(self.image_list[0]).shape[:2])
			self.gt_list = sorted(glob.glob(self.gt_path + '/*.h5'))

		self.transform = transform
		self.mode = mode
		self.scaling = scaling

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		image = cv2.imread(self.image_list[idx])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		gt = h5py.File(self.gt_list[idx], 'r')
		gt = np.asarray(gt['density'])

		rig = None  # 数据增强操作记录
		if self.mode == 'train':
			image, gt, rig = self._data_enhancement(image, gt)
		rgb = image.copy()

		if cfg.DEBUG:
			cv2.imwrite(f'./debug/image_{cfg.TIME}.png', image)
			cv2.imwrite(f'./debug/gt_{cfg.TIME}.png', gt * 255)

		if self.transform is not None:
			image = self.transform(image.copy())  # todo: 了解这里为什么如果不使用copy的话偶尔会报负步长的错误
		else:
			image = transforms.ToTensor()(image.copy())
			image = image / 255

		info = self.mode + '_' + os.path.basename(self.image_list[idx]).split('_')[1].split('.')[0] + f'_{rig}'

		return {
			'image': image,
			'gt': gt,
			'info': info,
			'rgb': rgb,  # todo: 分为训练和测试返回不同的内容
		}

	def _data_enhancement(self, image, gt):
		"""
		数据增强
		"""
		ratio = 0.5
		h, w = image.shape[:2]
		rdn = random.random()
		if rdn < 0.25:
			rig = '0_0'
			dx = 0
			dy = 0
		elif rdn < 0.5:
			rig = '1_0'
			dx = w * ratio
			dy = 0
		elif rdn < 0.75:
			rig = '0_1'
			dx = 0
			dy = h * ratio
		else:
			rig = '1_1'
			dx = w * ratio
			dy = h * ratio

		dx = int(dx)
		dy = int(dy)
		crop_x = int(dx + w * ratio)
		crop_y = int(dy + h * ratio)
		crop_image = image[dy:crop_y, dx:crop_x, :]
		crop_gt = gt[dy:crop_y, dx:crop_x]

		if random.random() > 0.8:
			rig += '_fliplr'
			crop_image = np.fliplr(crop_image)
			crop_gt = np.fliplr(crop_gt)

		# 对gt密度图进行缩放，但sum并不发生改变(0.0x误差)。缩放是为了提高网络的感受野，并且符合网络输出形状
		if self.scaling is not None and not cfg.DEBUG:
			crop_gt = cv2.resize(crop_gt, (int(crop_gt.shape[1] / 8), int(crop_gt.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * self.scaling ** 2

		return crop_image, crop_gt, rig
