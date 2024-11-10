import os
import glob
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from dataset import CustomDataset
from network.can import CANNet
from utils import save_model, load_model, save_density_image, save_image_with_contours, merge_density


def main(mode):
	start_epoch = 1
	# load model
	model = CANNet().to('cuda')
	if cfg.DATA.USE_CKPT:
		if cfg.DATA.CKPT_DATA is not None:
			path = sorted(glob.glob(cfg.DATA.CKPT_SAVE_PATH + f'/{cfg.TRAIN.CKPT_DATA}/*.pth'), key=os.path.getmtime)[
				-1]
		elif cfg.DATA.CKPT_NAME is not None:
			paths = sorted(glob.glob(cfg.DATA.CKPT_SAVE_PATH + '/**/*.pth'), key=os.path.getmtime)
			path = [path for path in paths if f'{cfg.TRAIN.CKPT_NAME}' in path][-1]  # 获取指定名字的模型
		else:
			path = sorted(glob.glob(cfg.DATA.CKPT_SAVE_PATH + '/**/*.pth'), key=os.path.getmtime)[-1]  # newest
		load_model(path, model)
		start_epoch = int(path.split('\\')[-1].split('_')[1]) + 1

	criterion = torch.nn.MSELoss(reduction='sum')  # loss sum
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

	best_mae = 1e10
	for epoch in tqdm(range(start_epoch, cfg.TRAIN.EPOCHS + 1)):
		if mode == 'both' or mode == 'train':
			train(model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)
		if mode == 'both' or mode == 'valid':
			pred_mae = valid(model=model, epoch=epoch)

			print(f'epoch: {epoch}\tpred_mae: {pred_mae:.3f}\tbest_mae: {best_mae:.3f}')
			if pred_mae < best_mae:
				best_mae = pred_mae

		if epoch % cfg.DATA.CKPT_SAVE_EPOCH == 0:
			save_name = f'ckpt_{epoch}_{cfg.CURRENT_TIME}'
			save_model(path=cfg.DATA.CKPT_SAVE_PATH + f'/{cfg.DATE_TIME}', model=model, name=save_name)


def train(model, criterion, optimizer, epoch):
	total_loss = 0
	model.train()

	# load dataset
	train_dataset: Any = DataLoader(CustomDataset(root=os.path.join(cfg.DATA.ROOT, 'train_data'),
	                                              mode='train',
	                                              scaling=cfg.DATA.SCALING,
	                                              transform=cfg.DATA.TRANSFORM),  # todo: 这里使用transformer会导致output为nan
	                                batch_size=cfg.DATA.BATCH_SIZE,
	                                shuffle=True)

	for i, data in enumerate(train_dataset):
		img = data['image'].to('cuda', dtype=torch.float32)
		tgt = data['gt'].to('cuda', dtype=torch.float32)
		info = data['info']

		output = model(img).squeeze()

		loss = criterion(output, tgt)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		with torch.no_grad():
			total_loss += loss.item()

		if i % cfg.TRAIN.LOG == 0:
			print(f'epoch:[{epoch}][{i}/{len(train_dataset)}]\t'
			      f'loss:{loss.item():.3f}->avg:{total_loss / (i + 1):.3f}\t'
			      f'pred:{torch.sum(output).item():.3f}->tgt:{torch.sum(tgt).item():.3f}')

	epoch_loss = total_loss / len(train_dataset)
	print(f'epoch: {epoch} loss: {epoch_loss:.3f}')


def valid(model, epoch):
	print('valid...')
	valid_dataset: Any = DataLoader(CustomDataset(root=os.path.join(cfg.DATA.ROOT, 'test_data'),
	                                              mode='valid',
	                                              transform=cfg.DATA.TRANSFORM),
	                                batch_size=1,
	                                shuffle=False)

	model.eval()
	mae = 0

	for i, data in enumerate(valid_dataset):
		img = data['image']
		tgt = data['gt']
		info = data['info'][0]
		rgb = data['rgb'][0]

		h, w = img.shape[2:4]
		h_biset = int(h / 2)
		w_biset = int(w / 2)

		img_1 = img[:, :, :h_biset, :w_biset].to('cuda', dtype=torch.float32)  # 0-0
		img_2 = img[:, :, h_biset:, :w_biset].to('cuda', dtype=torch.float32)  # 0-1
		img_3 = img[:, :, :h_biset, w_biset:].to('cuda', dtype=torch.float32)  # 1-0
		img_4 = img[:, :, h_biset:, w_biset:].to('cuda', dtype=torch.float32)  # 1-1

		density_1 = model(img_1).squeeze().detach().cpu().numpy()
		density_2 = model(img_2).squeeze().detach().cpu().numpy()
		density_3 = model(img_3).squeeze().detach().cpu().numpy()
		density_4 = model(img_4).squeeze().detach().cpu().numpy()

		pred_density = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
		tgt = tgt.squeeze().detach().cpu().numpy()
		mae += abs(pred_density - tgt.sum())

		if epoch % cfg.DATA.SAVE_IMAGE_EPOCH == 0:
			current = cfg.CURRENT_TIME
			merge = merge_density([density_1, density_2, density_3, density_4])
			binary_x = save_density_image(path=f'{cfg.DATA.SAVE_DENSITY_PATH}/{cfg.DATE_TIME}/{epoch}/density_{info}_{current}.png',
			                              x=merge, binarization=True, up_sample=8)
			if binary_x is not None:
				save_image_with_contours(path=f'{cfg.DATA.SAVE_IMAGE_PATH}/{cfg.DATE_TIME}/{epoch}/image_{info}_{current}.png',
				                         binary_x=binary_x, rgb=rgb)

	mae = mae / len(valid_dataset)

	return mae


if __name__ == '__main__':
	main(mode='both')
