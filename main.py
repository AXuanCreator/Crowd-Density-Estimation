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

	# optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

	# for p2p optimizer
	model_without_ddp = model
	param_dicts = [
		{"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
		{
			"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
			"lr": 1e-5,
		},
	]
	optimizer = torch.optim.Adam(param_dicts, lr=1e-4)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3500)

	# train & test
	best_mae = 1e10
	for epoch in tqdm(range(start_epoch, cfg.TRAIN.EPOCHS + 1)):
		if mode == 'both' or mode == 'train':
			train(model=model, net=net, criterion=criterion, optimizer=optimizer, epoch=epoch)
			lr_scheduler.step() if net == 'p2p' else None
		if mode == 'both' or mode == 'valid':
			# try:
			# 	pred_mae = valid(model=model, net=net, epoch=epoch)
			# except Exception as e:
			# 	print('error with valid: ', e)
			pred_mae = valid(model=model, net=net, epoch=epoch)

			print(f'epoch: {epoch}\tpred_mae: {pred_mae:.3f}\tbest_mae: {best_mae:.3f}')
			if pred_mae < best_mae:
				best_mae = pred_mae

		if epoch % cfg.DATA.CKPT_SAVE_EPOCH == 0:
			save_name = f'ckpt_{epoch}_{cfg.CURRENT_TIME.replace("-", "")}'
			save_model(path=cfg.DATA.CKPT_SAVE_PATH + f'/{cfg.DATE_TIME}/{net}', model=model, name=save_name)


def train(model, net, criterion, optimizer, epoch):
	model.train()
	total_loss = 0
	train_dataset: Any = DataLoader(
		CustomDataset(root=os.path.join(cfg.DATA.ROOT, 'train_data'), mode='train', net=net, scaling=cfg.DATA.SCALING, transform=cfg.DATA.TRANSFORM),
		batch_size=cfg.DATA.BATCH_SIZE, shuffle=True
	)

	for i, data in enumerate(train_dataset):
		img = data['image'].to('cuda', dtype=torch.float32)
		gt = {k: v.to('cuda', dtype=torch.float32) for k, v in data['gt'].items()} if isinstance(data['gt'], dict) else \
			data['gt'].to('cuda', dtype=torch.float32)

		output = model(img).squeeze() if isinstance(model(img), Tensor) else model(img)
		optimizer.zero_grad()

		if net in ['can', 'can-alex']:
			loss = criterion(output, gt)
		elif net == 'p2p':
			loss_dict = criterion(output, gt)
			loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
		else:
			raise NotImplementedError

		loss.backward()
		optimizer.step()
		total_loss += loss.item()

		if i % cfg.TRAIN.LOG == 0:
			pred_cnt = int((torch.nn.functional.softmax(output['pred_logits'], -1)[:, :, 1][
				                0] > 0.5).sum()) if net == 'p2p' else torch.sum(output).item()
			tgt_cnt = torch.sum(gt['labels']).item() if net == 'p2p' else torch.sum(gt).item()
			print(f'epoch:[{epoch}][{i}/{len(train_dataset)}]\tloss:{loss.item():.3f}->avg:{total_loss / (i + 1):.3f}\tpred:{pred_cnt:.3f}->tgt:{tgt_cnt:3f}')

	print(f'epoch: {epoch} loss: {total_loss / len(train_dataset):.3f}')


def valid(model, net, epoch):
	print('valid...')
	valid_dataset: Any = DataLoader(
		CustomDataset(root=os.path.join(cfg.DATA.ROOT, 'test_data'), mode='valid', net=net, transform=cfg.DATA.TRANSFORM),
		batch_size=1, shuffle=False
	)

	model.eval()
	mae = 0

	for i, data in enumerate(valid_dataset):
		img = data['image'].to('cuda', dtype=torch.float32)
		gt = {k: v.to('cuda', dtype=torch.float32) for k, v in data['gt'].items()} if isinstance(data['gt'], dict) else \
			data['gt'].to('cuda', dtype=torch.float32)
		info, rgb = data['info'][0], data['rgb'][0]

		h, w = img.shape[2:4]
		h_biset, w_biset = h // 2, w // 2
		img_parts = [
			img[:, :, :h_biset, :w_biset],
			img[:, :, h_biset:, :w_biset],
			img[:, :, :h_biset, w_biset:],
			img[:, :, h_biset:, w_biset:]
		]
		output_list = [model(img_part) for img_part in img_parts]

		if net in ['can', 'can-alex']:
			pred_cnt = sum(item.squeeze().detach().cpu().numpy().sum() for item in output_list)
			tgt_cnt = torch.sum(gt).item()
		elif net == 'p2p':
			outputs_scores_list = [torch.nn.functional.softmax(d['pred_logits'], -1)[:, :, 1][0] for d in output_list]
			pred_index_list = [(scores.detach().cpu().numpy() > 0.5) for scores in outputs_scores_list]
			pred_cnt = sum(sum(pred_index_list)).item()
			tgt_cnt = torch.sum(gt['labels']).item()
		else:
			raise NotImplementedError

		mae += abs(pred_cnt - tgt_cnt)

		if epoch % cfg.DATA.SAVE_IMAGE_EPOCH == 0 and random.random() > 0.5:
			current = cfg.CURRENT_TIME.replace('-', '')
			if net in ['can', 'can-alex']:
				merge = merge_density([density.squeeze().detach().cpu().numpy() for density in output_list])
				binary_x = save_density_image(
					path=f'{cfg.DATA.SAVE_DENSITY_PATH}/{cfg.DATE_TIME}/{epoch}/density_{info}_{current}.png',
					x=merge, binarization=True, up_sample=8
				)
				if binary_x is not None:
					save_image_with_contours(
						path=f'{cfg.DATA.SAVE_IMAGE_PATH}/{cfg.DATE_TIME}/{epoch}/image_{info}_{current}.png',
						binary_x=binary_x, rgb=rgb
					)
			elif net == 'p2p':
				pred_coord_list = []  # 相对于完整rgb图的x-y坐标
				for idx, (pred_index, output) in enumerate(zip(pred_index_list, output_list)):
					pred_coord = output['pred_points'].squeeze().detach().cpu().numpy()[pred_index]
					pred_coord_list.append(pred_coord + [0 if idx < 2 else w_biset,
					                                     0 if idx % 2 == 0 else h_biset])  # 将1/4坐标映射到全局坐标
				pred_coord = np.concatenate(pred_coord_list, axis=0).astype(int)  # 合并坐标数组
				plot_points_on_rgb(
					coords=pred_coord, rgb=rgb,
					path=f'{cfg.DATA.SAVE_DENSITY_PATH}/{cfg.DATE_TIME}/{epoch}/density_{info}_{current}.png',
					color='red'
				)
			else:
				raise NotImplementedError

	return mae / len(valid_dataset)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', '-m', type=str, default='both', help='train/valid/both')
	parser.add_argument('--net', '-n', type=str, default='can-alex', help='models: can/can-alex/p2p')

	args = parser.parse_args()
	main(args.mode, args.net)
