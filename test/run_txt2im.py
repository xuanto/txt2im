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



def run_txt2img():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="settings for txt2im model")
    parser.add_argument("--type", "-t", help="birds or coco", default="cub")

    pass
    run_txt2img()





