import os
import glob
import random
from typing import Any

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm

from config import cfg
from dataset import CustomDataset
from models.can import CANet, CanAlexNet
from models.p2pnet import P2PNet, P2P_Loss
from utils import save_model, load_model, save_density_image, save_image_with_contours, merge_density, \
	plot_points_on_rgb


def main(mode, net):
	start_epoch = 1
	# load model
	if net == 'can':
		model = CANet().to('cuda')
	elif net == 'can-alex':
		model = CanAlexNet().to('cuda')
	elif net == 'p2p':
		model = P2PNet().to('cuda')
	else:
		raise NotImplementedError

	# loading checkpoints
	if cfg.DATA.USE_CKPT:
		if cfg.DATA.CKPT_DATA is not None:  # 选择特定日期的最新权重
			path = sorted(glob.glob(cfg.DATA.CKPT_SAVE_PATH + f'/{cfg.TRAIN.CKPT_DATA}/{net}/*.pth'),
			              key=os.path.getmtime)[-1]
		elif cfg.DATA.CKPT_NAME is not None:  # 选择特定关键词的最新权重
			paths = sorted(glob.glob(cfg.DATA.CKPT_SAVE_PATH + '/**/*.pth'), key=os.path.getmtime)
			path = [path for path in paths if f'{cfg.TRAIN.CKPT_NAME}' in path][-1]  # 获取指定名字的模型
		else:  # 选择最新权重
			path = sorted(glob.glob(cfg.DATA.CKPT_SAVE_PATH + '/**/*.pth'), key=os.path.getmtime)[-1]  # newest
		load_model(path, model)
		start_epoch = int(path.split('\\')[-1].split('_')[1]) + 1

	# loss function
	if net == 'can' or net == 'can-alex':
		criterion = torch.nn.MSELoss(reduction='sum')  # loss function
	elif net == 'p2p':
		criterion = P2P_Loss()
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

	# train & test
	best_mae = 1e10
	for epoch in tqdm(range(start_epoch, cfg.TRAIN.EPOCHS + 1)):
		if mode == 'both' or mode == 'train':
			train(model=model, net=net, criterion=criterion, optimizer=optimizer, epoch=epoch)
		if mode == 'both' or mode == 'valid':
			pred_mae = valid(model=model, net=net, epoch=epoch)

			print(f'epoch: {epoch}\tpred_mae: {pred_mae:.3f}\tbest_mae: {best_mae:.3f}')
			if pred_mae < best_mae:
				best_mae = pred_mae

		if epoch % cfg.DATA.CKPT_SAVE_EPOCH == 0:
			save_name = f'ckpt_{epoch}_{cfg.CURRENT_TIME.replace("-", "")}'
			save_model(path=cfg.DATA.CKPT_SAVE_PATH + f'/{cfg.DATE_TIME}/{net}', model=model, name=save_name)


def train(model, net, criterion, optimizer, epoch):
	total_loss = 0
	model.train()

	# load dataset
	train_dataset: Any = DataLoader(CustomDataset(root=os.path.join(cfg.DATA.ROOT, 'train_data'),
	                                              mode='train',
	                                              net=net,
	                                              scaling=cfg.DATA.SCALING,
	                                              transform=cfg.DATA.TRANSFORM),  # todo: 这里使用transformer会导致output为nan
	                                batch_size=cfg.DATA.BATCH_SIZE,
	                                shuffle=True)

	for i, data in enumerate(train_dataset):
		img = data['image'].to('cuda', dtype=torch.float32)
		# target
		if isinstance(data['gt'], Tensor):
			data['gt'] = data['gt'].to('cuda', dtype=torch.float32)
		else:
			# 此等情况为嵌套字典，在net=p2p时触发
			for key, value in data['gt'].items():
				data['gt'][key] = value.to('cuda', dtype=torch.float32)

		info = data['info']

		output: Any = model(img)
		# todo: 1：1
		if isinstance(output, Tensor):
			output = output.squeeze()

		optimizer.zero_grad()
		if net == 'can' or net == 'can-alex':
			loss = criterion(output, data['gt'])
		elif net == 'p2p':
			loss_dict = criterion(output, data['gt'])
			weight_dict = criterion.weight_dict
			loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
		else:
			raise NotImplementedError
		loss.backward()
		optimizer.step()

		with torch.no_grad():
			total_loss += loss.item()

			if i % cfg.TRAIN.LOG == 0:
				if net == 'p2p':  # todo: 封装p2p和can的，不要东一块西一块
					# 先将概率值通过softmax，然后筛选出第二个值(表示人)， 最后选择第一个batch
					# 通过与threshold(0.5)比较，将outputs_scores变为bool数组，再相加获取判断出人的数量
					# 这种方式实际上坐标可能对不准，只供参考
					outputs_scores = torch.nn.functional.softmax(output['pred_logits'], -1)[:, :, 1][0]
					pred_cnt = int((outputs_scores > 0.5).sum())  # threshold一般为0.5
					tgt_cnt = torch.sum(data['gt']['labels']).item()
				else:
					pred_cnt = torch.sum(output).item()
					tgt_cnt = torch.sum(data['gt']).item()

				print(f'epoch:[{epoch}][{i}/{len(train_dataset)}]\t'
				      f'loss:{loss.item():.3f}->avg:{total_loss / (i + 1):.3f}\t'
				      f'pred:{pred_cnt:.3f}->tgt:{tgt_cnt:3f}')

	epoch_loss = total_loss / len(train_dataset)
	print(f'epoch: {epoch} loss: {epoch_loss:.3f}')


def valid(model, net, epoch):
	print('valid...')
	valid_dataset: Any = DataLoader(CustomDataset(root=os.path.join(cfg.DATA.ROOT, 'test_data'),
	                                              mode='valid',
	                                              net=net,
	                                              transform=cfg.DATA.TRANSFORM),
	                                batch_size=1,
	                                shuffle=False)

	model.eval()
	mae = 0

	for i, data in enumerate(valid_dataset):
		img = data['image'].to('cuda', dtype=torch.float32)
		# target
		if isinstance(data['gt'], Tensor):
			data['gt'] = data['gt'].to('cuda', dtype=torch.float32)
		else:
			# 此等情况为嵌套字典，在net=p2p时触发
			for key, value in data['gt'].items():
				data['gt'][key] = value.to('cuda', dtype=torch.float32)

		info = data['info'][0]
		rgb = data['rgb'][0]

		h, w = img.shape[2:4]
		h_biset = int(h / 2)
		w_biset = int(w / 2)

		img_1 = img[:, :, :h_biset, :w_biset].to('cuda', dtype=torch.float32)  # 0-0
		img_2 = img[:, :, h_biset:, :w_biset].to('cuda', dtype=torch.float32)  # 0-1
		img_3 = img[:, :, :h_biset, w_biset:].to('cuda', dtype=torch.float32)  # 1-0
		img_4 = img[:, :, h_biset:, w_biset:].to('cuda', dtype=torch.float32)  # 1-1

		output_list = [model(img) for img in [img_1, img_2, img_3, img_4]]  # list-tensor

		if net == 'can' or net == 'can-alex':
			pred_cnt = sum(item.squeeze().detach().cpu().numpy().sum() for item in output_list)
			tgt_cnt = torch.sum(data['gt']).item()
		elif net == 'p2p':
			outputs_scores_list = [torch.nn.functional.softmax(d['pred_logits'], -1)[:, :, 1][0] for d in output_list]
			pred_index_list = [(outputs_scores > 0.5) for outputs_scores in outputs_scores_list]  # threshold一般为0.5
			pred_cnt = sum([p.sum() for p in pred_index_list]).item()
			tgt_cnt = torch.sum(data['gt']['labels']).item()
		else:
			raise NotImplementedError

		mae += abs(pred_cnt - tgt_cnt)

		if epoch % cfg.DATA.SAVE_IMAGE_EPOCH == 0 and random.random() > 0.5:
			current = cfg.CURRENT_TIME.replace('-', '')
			if net == 'can' or net == 'can-alex':
				# 仅适用于输出为密度图的情况
				merge = merge_density([density.squeeze().detach().cpu().numpy() for density in output_list])
				binary_x = save_density_image(path=f'{cfg.DATA.SAVE_DENSITY_PATH}/{cfg.DATE_TIME}/{epoch}/density_{info}_{current}.png',
				                              x=merge, binarization=True, up_sample=8)
				if binary_x is not None:
					save_image_with_contours(path=f'{cfg.DATA.SAVE_IMAGE_PATH}/{cfg.DATE_TIME}/{epoch}/image_{info}_{current}.png',
					                         binary_x=binary_x, rgb=rgb)
			elif net == 'p2p':
				# 仅适用于p2p
				# 筛选出threshold>0.5的点的索引，然后再从coords筛选，最后在图片(rgb)上打点
				pred_coord_list = []  # 全图xy坐标
				for idx, _ in enumerate(pred_index_list):
					pred_index = pred_index_list[idx].cpu().numpy()
					pred_coord = output_list[idx]['pred_points'].squeeze().detach().cpu().numpy()[pred_index]
					if idx == 0:
						pred_coord_list.append(pred_coord)
					elif idx == 1:
						pred_coord_list.append(pred_coord + [0, h_biset])  # 左下, x * 1, y * 2
					elif idx == 2:
						pred_coord_list.append(pred_coord + [w_biset, 0])  # 右上, x * 2, y * 1
					else:
						pred_coord_list.append(pred_coord + [w_biset, h_biset])  # 右下, x * 2, y * 2
				pred_coord = np.array(pred_coord_list).reshape(-1, 2).astype(int)  # [ndarray] coords x-y
				plot_points_on_rgb(coords=pred_coord, rgb=rgb,
				                   path=f'{cfg.DATA.SAVE_DENSITY_PATH}/{cfg.DATE_TIME}/{epoch}/density_{info}_{current}.png',
				                   color='red')
			else:
				raise NotImplementedError

	mae = mae / len(valid_dataset)

	return mae


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', '-m', type=str, default='both', help='train/valid/both')
	parser.add_argument('--net', '-n', type=str, default='p2p', help='models: can/can-alex')

	args = parser.parse_args()
	main(args.mode, args.net)
