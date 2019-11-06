import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# # import argparse

from tqdm import trange
from models import *
from datasets import TextDataset,prepare_data

from cfg.config import cfg as cfgs


def train():
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
