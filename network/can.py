import cv2
import torchvision.models as models
from torch import nn
from torch.nn import functional as F

from .context import CANContext


class CANNet(nn.Module):
	def __init__(self, use_ckpt=False):
		super(CANNet, self).__init__()

		self.frontend_layer = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
		self.backend_layer = [512, 512, 512, 256, 128, 64]

		self.frontend = make_layers(self.frontend_layer)
		self.context = CANContext(feat_channels=self.frontend_layer[-1], out_channels=self.backend_layer[0]).to('cuda')
		self.backend = make_layers(self.backend_layer, in_channels=self.backend_layer[
			0], batch_norm=True, dilation=True)
		self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

		if not use_ckpt:
			self.__weight_init()

			vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
			vgg16_weights = list(vgg16.state_dict().items())
			frontend_weights = list(self.frontend.state_dict().items())
			for i in range(len(self.frontend.state_dict().items())):
				frontend_weights[i][1].data[:] = vgg16_weights[i][1].data[:]

	def forward(self, x):
		x = self.frontend(x)
		x = self.context(x)
		x = self.backend(x)

		return self.output_layer(x)

	def __weight_init(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight, std=0.01)  # mark: 这里不可以使用kaiming_normal_，会导致梯度爆炸
				if m.bias is not None:
					nn.init.zeros_(m.bias)


def make_layers(layer, in_channels=3, batch_norm=False, dilation=False):
	if dilation:
		d_rate = 2
	else:
		d_rate = 1
	layers = []
	for ly in layer:
		if ly == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			layers += [nn.Conv2d(in_channels, ly, kernel_size=3, padding=d_rate, dilation=d_rate)]
			if batch_norm:
				layers += [nn.BatchNorm2d(ly), nn.ReLU(inplace=True)]
			else:
				layers += [nn.ReLU(inplace=True)]
			in_channels = ly

	return nn.Sequential(*layers)
