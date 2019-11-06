import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.embeding_dim = 256
        
__C.channels = 64
__C.cuda_id = 1
__C.c_mod = "ip"
__C.hinge_loss = True
__C.condition_dim = 256
__C.z_dim = 20
__C.dk = 1
__C.image_size = 128
__C.lr_g = 1e-4
__C.lr_d = 1e-4

        
__C.batch_size = 86
__C.sentence_len = 18
        
# dataset config
__C.dataset = "coco" # "birds"
if __C.dataset == "coco":
    __C.text_encoder_file = "../data/coco/text_encoder100.pth"
    __C.train_epoch = 2000
    __C.save_model_dict = lambda x : x > 50 and x % 10 == 1
    __C.sampling = lambda x : x > 200 and x % 25 == 1
            # __C.save_model_dict = __C.sampling
    __C.num_words = 27297
else:
    __C.text_encoder_file = "../DAMSMencoders/bird/text_encoder200.pth"
    __C.train_epoch = 10000
    __C.save_model_dict = lambda x : x > 500 and x % 10 == 1
    __C.sampling = lambda x : x > 1000 and x % 50 == 1
    __C.num_words = 5450

__C.load_model = None    # "./logs/0424C1cocoS256/ep211/ep211"

# visualize pramaters
__C.tb_path = "./logs/0525C%d%sS%d" % (__C.cuda_id, __C.dataset, __C.image_size)
__C.save_path = __C.tb_path
__C.draw_after_iters = 5000
__C.tb_step = 30

# RNN settings
__C.DATASET_NAME = 'birds'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 6

__C.RNN_TYPE = 'LSTM'   # 'GRU'
__C.B_VALIDATION = False

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = '../models/bird_AttnGAN2.pth'
__C.TRAIN.NET_G = ''
__C.TRAIN.B_NET_D = True

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0


# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False


__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 18

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    #print(a.__dir__)
    for k, v in a.items():
        # a must specify keys that are in b
        #if not b.has_key(k):
        #    raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    
    Arguments:
        filename {[str]} 
    """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
