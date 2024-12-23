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

"""
todo:
dataset需要有标签0/1和其对应的x、y坐标
"""


class CustomDataset(Dataset):
	def __init__(self, root, mode, net, scaling=None, transform=None):
		self.root = root
		self.mode = mode
		self.net = net
		self.image_path = os.path.join(self.root, 'images')
		self.gt_path = os.path.join(self.root, 'ground_truth')

		self.image_list = sorted(glob.glob(self.image_path + '/*.jpg'))

		if self.net == 'can' or self.net == 'can-alex':
			self.gt_list = sorted(glob.glob(self.gt_path + '/*can*.h5'))
			if len(self.gt_list) == 0:
				print('generate .h5 from .mat (guassian filter)')
				generate_h5(self.gt_path, *cv2.imread(self.image_list[0]).shape[:2], net='can')
				self.gt_list = sorted(glob.glob(self.gt_path + '/*can*.h5'))
		elif self.net == 'p2p':
			self.gt_list = sorted(glob.glob(self.gt_path + '/*p2p*.h5'))
			if len(self.gt_list) == 0:
				print('generate .h5 from .mat (no filter)')
				generate_h5(self.gt_path, *cv2.imread(self.image_list[0]).shape[:2], net='p2p')
				self.gt_list = sorted(glob.glob(self.gt_path + '/*p2p*.h5'))
		else:
			assert self.net is None, 'net provision required'

		self.transform = transform
		self.scaling = scaling

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		image = cv2.imread(self.image_list[idx])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		gt = h5py.File(self.gt_list[idx], 'r')
		gt = np.asarray(gt['density'])

		# 下采样防止显存溢出
		if self.net == 'p2p':  # todo: 优化，将can和p2p模块化，而不是东一处西一处
			image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

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

		# image的序号名
		info = self.mode + '_' + os.path.basename(self.image_list[idx]).split('_')[1].split('.')[0] + f'_{rig}'

		if self.net == 'can' or self.net == 'can-alex':
			batch = {
				'image': image,
				'gt': gt,
				'info': info,
				'rgb': rgb,  # todo: 分为训练和测试返回不同的内容
			}
		elif self.net == 'p2p':
			# gt需要额外处理，添加反例
			labels = gt.flatten()  # 展平
			points = np.indices(gt.shape).reshape(2, -1).T  # gt_index对应的坐标
			gt = {
				'labels': labels,
				'points': points,
			}
			batch = {
				'image': image,
				'gt': gt,
				'info': info,
				'rgb': rgb,  # todo: 分为训练和测试返回不同的内容
			}
		else:
			assert self.net is None, 'net provision required'

		return batch

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
		# p2p模式无需缩放
		if self.scaling is not None and not cfg.DEBUG and self.net != 'p2p':
			crop_gt = cv2.resize(crop_gt, (
			int(crop_gt.shape[1] / 8), int(crop_gt.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * self.scaling ** 2

		return crop_image, crop_gt, rig
