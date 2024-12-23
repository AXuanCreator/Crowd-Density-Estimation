from itertools import chain

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

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
		regression = regression.permute(0, 2, 3, 1).contiguous().view(
			regression.shape[0], -1, 2)  # [batch_size, points_num, (x,y)]
		anchor_points = self.anchor_points(x_raw).repeat(P4_X.shape[0], 1, 1)  # 点的分数? 形状与regression相同 todo: P4_X可否换成x
		output_coord = regression + anchor_points

		# classification
		classification = self.classifier(P4_X)
		classification = classification.permute(0, 2, 3, 1)
		batch_size, width, height, _ = classification.shape
		classification = classification.view(batch_size, width, height, 4, 2).contiguous().view(P4_X.shape[0], -1, 2)

		return {
			'pred_logits': classification,  # 每个点都会有两个类别分数，用于表示该点为人/非人的概率
			'pred_points': output_coord,  # 点坐标
		}


class P2P_Loss(nn.Module):
	def __init__(self):
		""" Create the criterion.
		Parameters:
			num_classes: number of object categories, omitting the special no-object category
			matcher: module able to compute a matching between targets and proposals
			weight_dict: dict containing as key the names of the losses and as values their relative weight.
			eos_coef: relative classification weight applied to the no-object category
			losses: list of all the losses to be applied. See get_loss for list of available losses.
		"""
		super().__init__()
		self.num_classes = 1
		self.matcher = HungarianMatcher_Crowd(cost_class=1, cost_point=0.05)
		self.weight_dict = {'loss_ce': 1, 'loss_points': 0.0002}
		self.eos_coef = 0.5
		self.losses = ['labels', 'points']

		empty_weight = torch.ones(self.num_classes + 1, device='cuda')
		empty_weight[0] = self.eos_coef
		self.register_buffer('empty_weight', empty_weight)

	def loss_labels(self, outputs, targets, indices, num_points):
		"""
		对分类结果进行损失计算——CrossEntropy
		Classification loss (NLL)
		targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		"""
		assert 'pred_logits' in outputs
		src_logits = outputs['pred_logits']

		idx = self._get_src_permutation_idx(indices)
		target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets['labels'], indices)]).to(dtype=torch.int64)  # gt
		target_classes = torch.full(src_logits.shape[:2], 0,
		                            dtype=torch.int64, device=src_logits.device)
		target_classes[idx] = target_classes_o

		loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
		losses = {'loss_ce': loss_ce}

		return losses

	def loss_points(self, outputs, targets, indices, num_points):
		"""
		对点位置进行损失计算——MSE
		Args:
			outputs:
			targets:
			indices:
			num_points:

		Returns:

		"""
		assert 'pred_points' in outputs
		idx = self._get_src_permutation_idx(indices)
		src_points = outputs['pred_points'][idx]
		target_points = torch.cat([t[i] for t, (_, i) in zip(targets['points'], indices)], dim=0)

		loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

		losses = {}
		losses['loss_point'] = loss_bbox.sum() / num_points

		return losses

	def _get_src_permutation_idx(self, indices):
		"""

		Args:
			indices:

		Returns:

		"""
		# permute predictions following indices
		batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
		src_idx = torch.cat([src for (src, _) in indices])
		return batch_idx, src_idx

	def _get_tgt_permutation_idx(self, indices):
		# permute targets following indices
		batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
		tgt_idx = torch.cat([tgt for (_, tgt) in indices])
		return batch_idx, tgt_idx

	def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
		loss_map = {
			'labels': self.loss_labels,
			'points': self.loss_points,
		}
		assert loss in loss_map, f'do you really want to compute {loss} loss?'
		return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

	def _is_dist_avail_and_initialized(self):
		if not dist.is_available():
			return False
		if not dist.is_initialized():
			return False
		return True

	def _get_world_size(self):
		if not self._is_dist_avail_and_initialized():
			return 1
		return dist.get_world_size()

	def forward(self, outputs, targets):
		""" This performs the loss computation.
		Parameters:
			 outputs: dict of tensors, see the output specification of the model for the format
			 targets: list of dicts, such that len(targets) == batch_size.
					  The expected keys in each dict depends on the losses applied, see each loss' doc
		"""
		output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

		indices1 = self.matcher(output1, targets)

		num_points = sum(len(t) for t in targets['labels'])
		num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
		if self._is_dist_avail_and_initialized():
			torch.distributed.all_reduce(num_points)
		num_boxes = torch.clamp(num_points / self._get_world_size(), min=1).item()

		losses = {}
		for loss in self.losses:
			losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

		return losses


class HungarianMatcher_Crowd(nn.Module):
	"""
	Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
	"""

	def __init__(self, cost_class: float = 1, cost_point: float = 1):
		super().__init__()
		self.cost_class = cost_class
		self.cost_point = cost_point
		assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

	@torch.no_grad()
	def forward(self, outputs, targets):
		# 获取batch_size和检测点数量
		bs, num_queries = outputs["pred_logits"].shape[:2]

		# 将分类对数和点坐标展平(batch_size合并)，分类对数将经过Softmax处理
		out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
		out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

		# 拼接目标标签和点坐标，使其与out_prob和out_points相对应
		# tgt_ids = torch.cat([v["labels"] for v in targets])
		# tgt_points = torch.cat([v["point"] for v in targets])
		tgt_ids = torch.cat([v for v in targets['labels']]).to(dtype=torch.long)
		tgt_points = torch.cat([v for v in targets['points']])

		# 计算分类成本，使用负的概率值
		cost_class = -out_prob[:, tgt_ids]

		# 计算点之间的L2距离成本
		cost_point = torch.cdist(out_points, tgt_points, p=2)

		# 计算最终成本
		C = self.cost_point * cost_point + self.cost_class * cost_class
		C = C.view(bs, num_queries, -1).cpu()

		# 获取每个目标大小
		sizes = [len(v) for v in targets['points']]
		# 使用先行分配算法匹配每个批次的元素
		indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
		return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
