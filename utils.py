import os
import glob

import cv2
import torch
import h5py
import numpy as np
import scipy.io as io
from scipy.ndimage import gaussian_filter


def generate_h5(gt_path, h, w):
	gt_mat_path = sorted(glob.glob(gt_path + '/*.mat'))
	for gmp in gt_mat_path:
		mat = io.loadmat(gmp)
		mat = mat['image_info'][0, 0][0, 0][0]
		k = np.zeros((h, w))

		for i in range(len(mat)):
			mat_x = int(mat[i][0])
			mat_y = int(mat[i][1])
			if mat_x < w and mat_y < h:
				k[mat_y][mat_x] = 1

		k = gaussian_filter(k, 15)  # 高斯滤波会让值"散开"，但总值(sum)几乎不变。此举是为了提高神经网络的的容错率，加快收敛

		with h5py.File(gmp.replace('mat', 'h5'), 'w') as hf:
			hf['density'] = k


def save_model(path, model, name):
	"""
	保存模型
	Args:
		path: [str] 保存路径
		model: [nn.Module] 模型
		name: [str] 模型保存名

	Returns: None | 写入
	"""
	print('save model...')

	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)

	save_name = f'{path}/{name}.pth'
	torch.save(model.state_dict(), save_name)


def load_model(path, model):
	"""
	保存模型
	Args:
		path: [str] 载入路径
		model: [nn.Module] 网络模型

	Returns: None
	"""
	print('load model...')

	state_dict = torch.load(path, weights_only=True)
	model.load_state_dict(state_dict)


def save_density_image(path, x, binarization=True, up_sample=None):
	"""
	保存密度图
	Args:
		path: [str] 保存路径
		x: [tensor&ndarray] 灰度图 [h, w]
		binarization: [bool] 图像是否二值化 [0&255]
		up_sample: [int] 图像是否上采样

	Returns: None/ndarray | 输出图像文件
	todo: 优化二值化算法
	"""
	assert len(x.shape) == 2, 'only tensor/ndarray with [h, w]'

	res = None

	if isinstance(x, torch.Tensor):
		x = x.detach().cpu().numpy()

	x = np.where(x < 0, 0, x)  # max(0, X)
	x = x * 255

	if up_sample is not None:
		x = cv2.resize(x, dsize=None, fx=up_sample, fy=up_sample, interpolation=cv2.INTER_LANCZOS4)

	if binarization:
		x = np.where(x > 5, 255, 0).astype(np.uint8)
		res = x

	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(path), exist_ok=True)

	cv2.imwrite(path, x)

	return res


def save_image_with_contours(path, binary_x, rgb):
	"""
	保存带有检测框的RGB图像
	Args:
		path: [str] 保存路径
		binary_x: [ndarray] 二值化密度图 [h, w]
		rgb: [tensor/ndarray] RGB图像 [h, w, 3]

	Returns: None | 输出图像文件
	todo: 优化检测框，二值化太极端了
	"""
	assert len(binary_x.shape) == 2 and len(rgb.shape) == 3, 'binary_x shape: [h, w] | rgb shape: [h, w, 3]'

	if isinstance(rgb, torch.Tensor):
		rgb = rgb.detach().cpu().numpy()

	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(path), exist_ok=True)

	rgb = rgb.copy()
	contours, _ = cv2.findContours(binary_x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		# 扩大框的尺寸
		h = h*(10-h) if h < 10 else h
		w = w*(10-w) if w < 10 else w
		cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)

	cv2.imwrite(path, rgb[:, :, [2, 1, 0]])  # imwrite需要bgr格式


def merge_density(X: list):
	"""
	将四等分的密度图合成为一张完整的图(灰度)
	输入顺序(x-y):
	0-0 -> 0-1 -> 1-0 -> 1-1
	Args:
		X: list[tensor/ndarray] len->4

	Returns: ndarray
	"""
	assert len(X) == 4, 'input length must be 4'

	X = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in X]

	top = np.hstack((X[0], X[2]))
	bottom = np.hstack((X[1], X[3]))
	result = np.vstack((top, bottom))

	return result
