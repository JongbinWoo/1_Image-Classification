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
    
    config.PATH.SAVEDIR = '/opt/ml/1_Image-Classification/checkpoint'
    # config.PATH.CHECKPOINT = args.r
    config.PATH.ROOT = '/opt/ml/input/data/train/images'
    config.PATH.TEST_1 = '/opt/ml/code/checkpoint/DenseNet_5'
    

    config.DATASET.NUM_CLASSES = [2, 3, 3]

    config.TRAIN.WIDTH = 256
    config.TRAIN.HEIGHT = 384

    config.TRAIN.EPOCH = 10 #args.epochs
    config.TRAIN.BATCH_SIZE = 64 #args.batch_size
    config.TRAIN.NUM_WORKERS = 4
    config.TRAIN.BASE_LR = 1e-4 #args.lr 
    config.TRAIN.PERIOD = 1
    config.TRAIN.RESUME = False
    config.TRAIN.KFOLD = 5
    config.TRAIN.LOSS = 'Cumbo'
    config.TRAIN.T = 10
    config.TRAIN.ALPHA = 0.05
    config.TRAIN.M = 1.3
    config.TRAIN.GAMMA = 0.5

    config.MODEL.PRETRAINED = True
    config.MODEL.FREEZE = False
    config.MODEL.OPTIM = 'Adam'
    config.MODEL.HIDDEN = 512
    
    

    return config



