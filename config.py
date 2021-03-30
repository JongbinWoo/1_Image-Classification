import torch

class AttributeDict(dict):
    def __init__(self):
        self.__dict__ = self

class ConfigTree:
    def __init__(self):
        self.SYSTEM = AttributeDict()
        self.PATH = AttributeDict()
        self.DATASET = AttributeDict()
        self.TRAIN = AttributeDict()
        self.MODEL = AttributeDict()
        self.KD = AttributeDict()

def get_config():
    config = ConfigTree()
    config.SYSTEM.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config.PATH.SAVEDIR = './checkpoint'
    # config.PATH.CHECKPOINT = args.r
    config.PATH.ROOT = '/opt/ml/input/data/train/images'
    config.PATH.RESUME = '/opt/ml/code/checkpoint/DENSE2_6'

    config.DATASET.NUM_CLASSES = [2, 3, 3]
    config.DATASET.RATIO = 0.3

    config.TRAIN.AUGMENTATION = {'size': 28,
                                 'use_flip': False,
                                 'use_color_jitter': False,
                                 'use_normalize': False}
                                 
    config.TRAIN.EPOCH = 50 #args.epochs
    config.TRAIN.BATCH_SIZE = 128 #args.batch_size
    config.TRAIN.NUM_WORKERS = 4
    config.TRAIN.BASE_LR = 0.0001 #args.lr 
    config.TRAIN.PERIOD = 3

    config.MODEL.OPTIM = 'Adam'
    config.MODEL.HIDDEN = [32, 64, 128]

    

    return config



