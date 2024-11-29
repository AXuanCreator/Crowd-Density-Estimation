from itertools import chain

import torch
import torch.nn as nn
import numpy as np

from .can import make_conv2d, make_layers


class AnchorPoints(nn.Module):
	def __init__(self, pyramid_levels=None, strides=None, row=2, line=2):
		super(AnchorPoints, self).__init__()

		if pyramid_levels is None:
			self.pyramid_levels = [3, 4, 5, 6, 7]
		else:
			self.pyramid_levels = pyramid_levels

		if strides is None:
			self.strides = [2 ** x for x in self.pyramid_levels]
		else:
			self.strides = strides

		self.row = row
		self.line = line

	def forward(self, image):
		# 获取图像的金字塔层级
		image_shape = np.array(image.shape[2:])
		image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

		all_anchor_points = np.zeros((0, 2), dtype=np.float32)

		for idx, p in enumerate(self.pyramid_levels):
			archor_points = self.generate_anchor_points(2 ** p)
			shifted_anchor_points = self.shift(image_shapes[idx], self.strides[idx], archor_points)
			all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

		all_anchor_points = np.expand_dims(all_anchor_points, axis=0).astype(np.float32)

		return torch.from_numpy(all_anchor_points).cuda()

	def generate_anchor_points(self, stride=16):
		row_step = stride / self.row
		line_step = stride / self.line

		shift_x = (np.arange(1, self.line + 1) - 0.5) * line_step - stride / 2
		shift_y = (np.arange(1, self.row + 1) - 0.5) * row_step - stride / 2

		shift_x, shift_y = np.meshgrid(shift_x, shift_y)

		anchor_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

		return anchor_points

	def shift(self, shape, stride, anchor_points):
		shift_x = (np.arange(0, shape[1]) + 0.5) * stride
		shift_y = (np.arange(0, shape[0]) + 0.5) * stride

		shift_x, shift_y = np.meshgrid(shift_x, shift_y)
		shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

		A = anchor_points.shape[0]
		K = shifts.shape[0]

		all_anchor_points = ((anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
		                     .reshape((K * A, 2)))

		return all_anchor_points


class P2PNet(nn.Module):
	def __init__(self):
		super(P2PNet, self).__init__()

		# vgg16 without fc
		self.vgg16_body12 = make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256], in_channels=3, batch_norm=True)
		self.vgg16_body3 = make_layers(['M', 512, 512, 512], in_channels=256, batch_norm=True)
		self.vgg16_body4 = make_layers(['M', 512, 512, 512], in_channels=512, batch_norm=True)

		# decoder
		self.decoder1 = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
			nn.Upsample(scale_factor=2, mode='nearest'),
		)
		# todo: 查看decoder2的in_channels是否符合规则
		self.decoder2 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
		self.decoder3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

		# regression
		self.regression = nn.Sequential(
			*list(chain(*[make_conv2d(256, 256, k_size=3, stride=1, padding=1) for _ in
			              range(2)])) + make_conv2d(256, 4 * 2, k_size=3, stride=1, padding=1)
		)
		# archor
		self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=2, line=2)

		# classification
		self.classifier = nn.Sequential(
			*list(chain(*[make_conv2d(256, 256, k_size=3, stride=1, padding=1) for _ in
			              range(2)])) + make_conv2d(256, 4 * 2, k_size=3, stride=1, padding=1, activation='sigmoid')
		)

	# todo: 添加预训练权重载入

	def forward(self, x):
		# todo: 将2和4替换为num_anchor_points和num_classes
		x_raw = x.clone()
		x = self.vgg16_body12(x)
		x_vgg_body3 = self.vgg16_body3(x)
		x_vgg_body4 = self.vgg16_body4(x_vgg_body3)

		P5_X = self.decoder1(x_vgg_body4)
		P4_X = self.decoder3(P5_X + self.decoder2(x_vgg_body3))

		# regression
		regression = self.regression(P4_X)
		regression = regression.permute(0, 2, 3, 1).contiguous().view(regression.shape[0], -1, 2)  # [batch_size, points_num, (x,y)]
		anchor_points = self.anchor_points(x_raw).repeat(P4_X.shape[0], 1, 1)  # 点的偏移量？ todo: P4_X可否换成x
		output_coord = regression + anchor_points

		# classification
		classification = self.classifier(P4_X)
		classification = classification.permute(0, 2, 3, 1)
		batch_size, width, height, _ = classification.shape
		classification = classification.view(batch_size, width, height, 4, 2).contiguous().view(P4_X.shape[0], -1, 2)

		return {
			'pred_logits': classification,
			'pred_points': output_coord,
		}
