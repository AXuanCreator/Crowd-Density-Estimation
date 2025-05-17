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