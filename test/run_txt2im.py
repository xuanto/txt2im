import os
import sys
import time
import random
import datetime
import dateutil.tz
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from tqdm import trange
from models import *
from datasets import TextDataset,prepare_data

import argparse



def run_txt2img(list_of_captions):
    z = torch.randn(batch_size, self.z_dim).to(self.cuda_id)
        
    fake_images = self.G(z, sent_emb, words_emb).detach()
    pass

    imgs , captions = batch
    batch_size = real_images.shape[0]
    words_emb = captions.cuda(self.cuda_id)
    sent_emb = torch.mean(captions,1).cuda(self.cuda_id)

    z = torch.randn(batch_size, self.z_dim).to(self.cuda_id)
    fake_images = self.G(z, sent_emb, words_emb).detach().cpu()

    for i, j in zip(fake_images, list_of_captions):
        image_file_name = "/%d.jpg" % j
        transforms.ToPILImage()(i).save(self.image_path + image_file_name)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="settings for txt2im model")
    parser.add_argument("--type", "-t", help="birds or coco", default="cub")

    pass
    run_txt2img()





