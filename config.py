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
    config.PATH.RESUME_1 = '/opt/ml/code/checkpoint/DenseNet_9'
    config.PATH.RESUME_2 = '/opt/ml/code/checkpoint/EfficientNet_0'
    config.PATH.TEST_1 = '/opt/ml/code/checkpoint/DenseNet_9'
    config.PATH.TEST_2 = '/opt/ml/code/checkpoint/EfficientNet_0'
    config.PATH.LOG = '/opt/ml/code/saved/log'

    config.DATASET.NUM_CLASSES = [2, 3, 3]
    config.DATASET.RATIO = 0.1

    config.TRAIN.AUGMENTATION = {'size': 28,
                                 'use_flip': False,
                                 'use_color_jitter': False,
                                 'use_normalize': False}
                                 
    config.TRAIN.EPOCH = 30 #args.epochs
    config.TRAIN.BATCH_SIZE = 64 #args.batch_size
    config.TRAIN.NUM_WORKERS = 4
    config.TRAIN.BASE_LR = 3e-4 #args.lr 
    config.TRAIN.PERIOD = 1
    config.TRAIN.RESUME = False

    config.MODEL.PRETRAINED = True
    config.MODEL.FREEZE = False
    config.MODEL.OPTIM = 'Adam'
    config.MODEL.HIDDEN = 512
    
    

    return config



