from datetime import datetime

from easydict import EasyDict as edict
from torchvision import transforms

'''todo
对不同网络使用不同的配置文件
'''
@property
def current_time(self): return datetime.now().strftime("%H-%M-%S")


__C = edict()
cfg = __C

__C.DATA = edict()
__C.TRAIN = edict()
__C.TEST = edict()

__C.DEBUG = False
__C.CURRENT_TIME = current_time.__get__(__C)  # 当前时间(动态更新), 用于文件命名
__C.DATE_TIME = datetime.now().strftime("%Y-%m-%d")  # 当日时间(按运行程序的日期), 用于文件夹命名
# -------------------------DATA--------------------------- #
__C.DATA.TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
__C.DATA.SAVE_IMAGE_EPOCH = 100
__C.DATA.CKPT_SAVE_EPOCH = 100  # 每_epoch保存权重

__C.DATA.ROOT = './dataset/ShanghaiTech_Crowd_Counting_Dataset/part_B_final'
__C.DATA.SAVE_IMAGE_PATH = './result/images'
__C.DATA.SAVE_DENSITY_PATH = './result/density'
__C.DATA.CKPT_SAVE_PATH = './result/ckpt/'  # 模型权重文件保存位置
__C.DATA.LOG_DIR = './logs/tensorboard'

__C.DATA.USE_CKPT = True  # 使用保存的模型
__C.DATA.CKPT_DATA = None  # or like '2024-11-11'
__C.DATA.CKPT_NAME = 'ckpt_500_160234'  # or like 'ckpt_40'

__C.DATA.BATCH_SIZE = 40
__C.DATA.SCALING = 8
# --------------------------TRAIN-------------------------- #
__C.TRAIN.SHUFFLE = True
__C.TRAIN.EPOCHS = 500
__C.TRAIN.PRETRAINED = False
__C.TRAIN.LR = 1e-4  # 1e-4 default
__C.TRAIN.WEIGHT_DECAY = 5 * 1e-4  # 5 * 1e-4 default
__C.TRAIN.LOG = 5
