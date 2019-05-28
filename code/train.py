import os
import sys
import time
import random
# import pprint
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

# from torchvision.datasets import CocoCaptions
import torchvision.transforms as transforms


from tqdm import trange
# from networks import Trainer
from models import *
from datasets import TextDataset,prepare_data


# dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
# sys.path.append(dir_path)

class Cfgs(object):
    """docstring for Cfgs"""
    def __init__(self):
        super(Cfgs, self).__init__()
        self.embeding_dim = 256
        
        self.channels = 64
        self.cuda_id = 1
        self.c_mod = "ip"
        self.hinge_loss = True
        # self.condit_decoder = True
        # self.big_dataset = True
        self.condition_dim = 256
        self.z_dim = 20
        self.dk = 1

        self.image_size = 128

        self.lr_g = (1e-4)
        self.lr_d = (1e-4)

        
        self.batch_size = 86
        self.sentence_len = 18
        
        # dataset config
        self.dataset = "coco"
        # self.dataset = "birds"
        if self.dataset == "coco":
            self.text_encoder_file = "../data/coco/text_encoder100.pth"
            self.train_epoch = 2000
            self.save_model_dict = lambda x : x > 50 and x % 10 == 1
            self.sampling = lambda x : x > 200 and x % 25 == 1
            # self.save_model_dict = self.sampling
            self.num_words = 27297
        else:
            self.text_encoder_file = "../DAMSMencoders/bird/text_encoder200.pth"
            self.train_epoch = 10000
            self.save_model_dict = lambda x : x > 500 and x % 10 == 1
            self.sampling = lambda x : x > 1000 and x % 50 == 1
            self.num_words = 5450

        self.load_model = None
        # self.load_model = "./logs/0424C1cocoS256/ep211/ep211"

        # visualize pramaters
        self.tb_path = "./logs/0525C%d%sS%d" % (self.cuda_id, self.dataset, self.image_size)
        self.save_path = self.tb_path
        self.draw_after_iters = 5000
        self.tb_step = 30

        # self.WORDS_NUM = 7263
        # self.RNN_TYPE = "GRU"


def train():
    cfgs = Cfgs()
    imsize = cfgs.image_size
    DATA_DIR = "../data/" + cfgs.dataset
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)), #]) #,
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip() ])
    ts =  transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    dataset = TextDataset(DATA_DIR, 'train',base_size=64,transform=image_transform,target_transform=ts)

    print(dataset.n_words, dataset.embeddings_num)

    #assert dataset
    dataloader = DataLoader(dataset, batch_size=cfgs.batch_size, drop_last=True,shuffle=True, num_workers=5)
    # trainer = condGANTrainer(dataloader,dataset.n_words)
    # # validation data #
    print(DATA_DIR)
    dataset_val = TextDataset(DATA_DIR, 'test',base_size=64,transform=image_transform,target_transform=ts)
    dataloader_val = DataLoader(dataset_val, batch_size=cfgs.batch_size, drop_last=True,shuffle=False, num_workers=5)
    print("****")


    print("init network ...")
    network = Trainer(cfgs)
    print("%d iters a epoch !!" % (len(dataset)//cfgs.batch_size))
    print('ad training start ! mod %s , at cuda%d !' % (cfgs.c_mod , cfgs.cuda_id))
    start_num = 0
    if cfgs.load_model is not None:
        print("loading model ...")
        network.load_model(cfgs.load_model) # ep120_iter117840_G.sd
        print("model loaded !! start training at ep %s" % cfgs.load_model)
        start_num = 211


    for epoch in trange(start_num, cfgs.train_epoch):
        #torch.cuda.empty_cache()
        if cfgs.save_model_dict(epoch):
            save_path = network.make_dir(epoch)
            print("saving ...")
            filename = "/ep%d" % (epoch) 
            network.save_model(save_path + filename)
            if cfgs.sampling(epoch):
                print("validating ...")
                for batch in dataloader_val:
                    network.validate(batch)
        else:
            for batch in dataloader:
                network.train_step(batch)
            # try:
            #     network.learning_rate_decay(0.5)
            # except:
            #     print("lrd error !!!")


def test():
    cfgs = Cfgs()
    imsize = cfgs.image_size
    DATA_DIR = "../data/coco"
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)), #]) #,
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip() ])
    ts =  transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    dataset = TextDataset(DATA_DIR, 'train',base_size=64,transform=image_transform,target_transform=ts)
    print("dataset.n_words:", dataset.n_words, "dataset.embeddings_num:", dataset.embeddings_num)

    dataloader = DataLoader(dataset, batch_size=cfgs.batch_size, drop_last=True,shuffle=True, num_workers=5)
    print(DATA_DIR)
    print("****")


    for batch in dataloader:
        print(type(batch))



if __name__ == '__main__':
    train()
    # test()
