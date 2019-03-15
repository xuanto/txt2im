from __future__ import print_function

# from miscc.utils import mkdir_p
# from miscc.utils import build_super_images
# from miscc.losses import sent_loss, words_loss
# from miscc.config import cfg #, cfg_from_file

from datasets import TextDataset,prepare_data

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
# import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchvision.datasets import CocoCaptions
import torchvision.transforms as T


from tqdm import trange
from models import *
from bert_serving.client import BertClient


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

class Cfgs(object):
    """docstring for Cfgs"""
    def __init__(self):
        super(Cfgs, self).__init__()
        # hp
        self.embeding_dim = 256
        
        self.channels = 64
        self.cuda_id = 1
        self.c_mod = "ip"
        self.condit_decoder = True
        self.big_dataset = True
        self.condition_dim = 768
        self.z_dim = 20
        self.dk = 1

        self.image_size = 128

        self.lr_g = (1e-4)
        self.lr_d = (1e-4)

        # dataset config
        self.batch_size = 32
        self.dataset = "cub" #"coco"
        if self.big_dataset:
            self.data_kind = "val"
            self.train_epoch = 3000
            self.draw_after_iters = 2000
            self.save_after_epochs = 10
            self.draw_after_epochs = 10
        else:
            self.data_kind = "val"
            self.train_epoch = 20000
            self.draw_after_iters = 1000
            self.save_after_epochs = 1000
            self.draw_after_epochs = 20
        # ll
        self.load_model = None
        # self.load_model = "./logs/0131C1S256BWAsHMipk1.0/ep8000_iter56000"

        #self.CLP = 0.02
        self.began = False
        self.residual = False
        # visualize pramaters
        self.normalize_images = False
        self.tb_path = "./logs/cocoC%dS%dM%sk%.1f" % (self.cuda_id, self.image_size, self.c_mod, self.dk)
        # self.tb_path = "./logs/0117Cheat" # % (self.cuda_id, self.image_size, self.c_mod, self.dk)


        self.save_path = self.tb_path

        self.WORDS_NUM = 7263
        self.RNN_TYPE = "GRU"


class Coco_collate_fn(object):
    """docstring for Coco_collate_fn"""
    def __init__(self):
        super(Coco_collate_fn, self).__init__()
        self.bc = BertClient(check_length=False)

    def __call__(self, batch):
        """
        Collate function to be used when wrapping CocoCaptions in a
        DataLoader. Returns a tuple of the following:

        - imgs: FloatTensor of shape (N, C, H, W)
        - captions: LongTensor of shape (O,) giving object categories
        """
        images, captions = [], []
        for i, (img, caps) in enumerate(batch):
            images.append(img)
            rand_idx = random.randint(0,len(caps)-1)
            captions.append(caps[0])
        captions = torch.from_numpy(self.bc.encode(captions))
        images = torch.stack(images)
        return images,captions


def train():
    cfgs = Cfgs()
    transform_im = [T.Resize((128,128)), T.ToTensor()]
    # if cfgs.normalize_images:
    #     transform_im.append(T.Lambda(lambda x : (x-127.5)/127.5))
    transform_im = T.Compose(transform_im)
# if cfgs.dataset == "coco":
    caproot = "../data/coco/annotations/captions_train2017.json"
    imgroot = "../data/coco/train2017"
    train_dset = CocoCaptions(root = imgroot,annFile = caproot,transform=transform_im)
    coco_collate_fn = Coco_collate_fn()
    dataloader = DataLoader(dataset=train_dset, batch_size=cfgs.batch_size, 
                        collate_fn=coco_collate_fn,shuffle=True, num_workers=0)
    
    caproot = "../data/coco/annotations/captions_val2017.json"
    imgroot = "../data/coco/val2017"
    train_dset_val = CocoCaptions(root = imgroot,annFile = caproot,transform=transform_im)
    coco_collate_fn = Coco_collate_fn()
    dataloader_val = DataLoader(dataset=train_dset_val, batch_size=cfgs.batch_size, 
                        collate_fn=coco_collate_fn,shuffle=True, num_workers=0)


    # imsize = cfgs.image_size
    # DATA_DIR = "../data/birds"
    # image_transform = transforms.Compose([
    #     transforms.Resize(int(imsize * 76 / 64)), #]) #,
    #     transforms.RandomCrop(imsize) ]) #,
    #     # transforms.RandomHorizontalFlip()])
    # ts =  transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    # dataset = TextDataset(DATA_DIR, 'train',base_size=64,transform=image_transform,target_transform=ts)

    # print(dataset.n_words, dataset.embeddings_num)

    # #assert dataset
    # dataloader = DataLoader(dataset, batch_size=cfgs.batch_size, drop_last=True,shuffle=True, num_workers=5)
    # # trainer = condGANTrainer(dataloader,dataset.n_words)
    # # # validation data #
    # print(DATA_DIR)
    # dataset_val = TextDataset(DATA_DIR, 'test',base_size=64,transform=image_transform,target_transform=ts)
    # dataloader_val = DataLoader(dataset_val, batch_size=cfgs.batch_size, drop_last=True,shuffle=False, num_workers=5)
    # print("****")


    print("init network ...")
    network = Trainer(cfgs)
    print("%d iters a epoch !!" % (len(train_dset)//cfgs.batch_size))
    print('ad training start ! mod %s , at cuda%d !' % (cfgs.c_mod , cfgs.cuda_id))
    start_num = 0
    if cfgs.load_model is not None:
        print("loading model ...")
        network.load_model(cfgs.load_model) # ep120_iter117840_G.sd
        print("model loaded !! start training at ep %s" % cfgs.load_model[:3])
        start_num = 3000

    for epoch in trange(start_num, cfgs.train_epoch):
        
        #torch.cuda.empty_cache()
        if epoch % cfgs.draw_after_epochs != cfgs.draw_after_epochs - 1:
            for batch in dataloader:
                network.train_step(batch)
        else:
            print("validating ...")
            network.make_dir()
            for batch in dataloader_val:
                network.validate(batch)

        
        # if epoch % cfgs.draw_after_epochs == 0:
        #     # rd_seed = random.randint(0,5)
        #     network.save_image(train_dset)
        if epoch % cfgs.save_after_epochs == cfgs.save_after_epochs - 1:
            filename = "/ep%d" % (epoch+1)
            try:
                network.save_model(cfgs.tb_path+filename)
            except:
                print("save_model error!!")
            try:
                network.learning_rate_decay(0.5)
            except:
                print("lrd error !!!")


if __name__ == '__main__':
    train()
