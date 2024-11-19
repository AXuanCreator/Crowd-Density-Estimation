import torch
from torch import nn, Tensor
from torch.nn import functional as F


class CANContext(nn.Module):
	def __init__(self, feat_channels, out_channels):
		super(CANContext, self).__init__()

		self.scales = (1, 2, 3, 6)
		self.scales_module = nn.ModuleList([nn.Sequential(
			nn.AdaptiveAvgPool2d((s, s)), nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=False))
			for s in self.scales])
		self.weight_net = nn.Sequential(
			nn.Conv2d(feat_channels, feat_channels, kernel_size=1),
			nn.Sigmoid(), )
		self.output_layer = nn.Sequential(
			nn.Conv2d(feat_channels*2, out_channels, kernel_size=1),
			nn.ReLU()
		)

	def forward(self, feat: Tensor):
		feat_context = [F.interpolate(s_mod(feat), size=feat.shape[2:4], mode='bilinear') for s_mod in self.scales_module]
		feat_weight = [self.weight_net(feat - f_c) for f_c in feat_context]

		features = [sum(feat_context[i] * feat_weight[i] for i in range(len(self.scales))) / sum(feat_weight)] + [feat]
		features = torch.cat(features, 1)  # channels * 2

		return self.output_layer(features)