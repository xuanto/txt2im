# -*- coding:utf-8 -*-
"""
file: models.py
"""
import json
import random
import os
#_______ torch _______
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn import init
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#_______ vision _______
from torchvision.utils import make_grid
from torchvision  import transforms  
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from cfg.config import cfg
from layers import *
from tensorboardX import SummaryWriter
# from GlobalAttention import GlobalAttentionGeneral as ATT_NET


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True,total_length=18)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb

class Generator(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cuda_id = cfgs.cuda_id
        self.z_dim = cfgs.z_dim
        self.st_dim = cfgs.condition_dim
        chn = cfgs.channels
        self.first_view = 16 * chn
        self.c_dim = self.st_dim + self.z_dim

        # self.T_linear = SpectralNorm(nn.Linear(cfgs.condition_dim, self.st_dim))
        self.frist_size = cfgs.image_size // 32
        self.G_linear = SpectralNorm(nn.Linear(self.z_dim, self.first_view * self.frist_size * self.frist_size))

        # # self.c_attn = ChannelWiseAttention(4*chn) 
        # self.g_attn = GlobalAttentionGeneral(1*chn, cfgs.condition_dim) 

        # self.CA = CA_NET(cfgs)
        # self.s_attn = SelfAttention(1*chn)
        self.conv = nn.ModuleList([ #GBlock(16*chn, 16*chn, condition_dim = self.c_dim),
                                    GBlock(16*chn, 8*chn, condition_dim = self.c_dim),
                                    GBlock(8*chn, 4*chn, condition_dim = self.c_dim),                                    
                                    GBlock(4*chn, 2*chn, condition_dim = self.c_dim),
                                    GBlock(2*chn, 1*chn, condition_dim = self.c_dim),
                                    GBlock(1*chn, 1*chn, condition_dim = self.c_dim)])

        # self.s_attn = SelfAttention(2*chn)
        # self.conv = nn.ModuleList([ #GBlock(16*chn, 16*chn, condition_dim = self.c_dim),
        #                             #GBlock(16*chn, 8*chn, condition_dim = self.c_dim),
        #                             GBlock(8*chn, 8*chn, condition_dim = self.c_dim),                                    
        #                             GBlock(8*chn, 4*chn, condition_dim = self.c_dim),
        #                             GBlock(4*chn, 2*chn, condition_dim = self.c_dim),
        #                             GBlock(2*chn, 1*chn, condition_dim = self.c_dim)])
        self.colorize = SpectralNorm(nn.Conv2d(1*chn, 3, [3, 3], padding=1))

    def forward(self, z, sentence_emb, words_emb):
        """
        Arguments:
            input {[type]} -- [description]
            sentence_emb {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        #codes = torch.split(z, self.z_dim, 1)  # 120 -> 5 x 20
        B = sentence_emb.shape[0]
        out = self.G_linear(z).view(B,self.first_view,self.frist_size,self.frist_size)
        # sentence_emb , mu, logvar = self.CA(sentence_emb)
        # out = self.G_linear(sentence_emb)

        for i, conv in enumerate(self.conv): 
            # if i == 3:
            #     out = self.s_attn(out)
                # out_sa = self.s_attn(out)
                # out_ga, g_attn = self.g_attn(out, words_emb)
                # out = torch.cat([out_sa, out_ga], 1)
            conv_z = torch.randn([B, self.z_dim]).to(self.cuda_id)
            condition = torch.cat([conv_z, sentence_emb], 1)  # ([B, 20 + 768])
            out = conv(out, condition)

        out = F.relu(out, True)
        out = self.colorize(out)
        return torch.tanh(out)

class Discriminator(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.condit_decoder = cfgs.condit_decoder
        self.chn = cfgs.channels
        self.c_dim = cfgs.condition_dim
        self.k = cfgs.dk
        chn = self.chn
        # self.CA = CA_NET(cfgs)
        self.pre_conv = nn.Sequential(SpectralNorm(nn.Conv2d(3, 1*chn, 3,padding=1),),
                                      nn.ReLU(True),
                                      SpectralNorm(nn.Conv2d(1*chn, 1*chn, 3,padding=1),),
                                      nn.AvgPool2d(2))

        self.pre_skip = SpectralNorm(nn.Conv2d(3, 1*chn, 1))
        #self.T_linear = SpectralNorm(nn.Linear(cfgs.condition_dim, self.c_dim))

        # self.c_attn = ChannelWiseAttention(chn*8)
        # self.s_attn = SelfAttention(1*chn)
        # self.s_attn = nn.Sequential(    GBlock(1*chn, 1*chn, downsample=True, bn=False, upsample=False),
        #                                 SelfAttention(1*chn) ,
        #                                 GBlock(1*chn, 2*chn, downsample=True, bn=False, upsample=False)
        #                                 )
        # self.g_attn = GlobalAttentionGeneral(1*chn, cfgs.condition_dim)

        self.encoder = nn.ModuleList([  #GBlock(1*chn, 1*chn, downsample=True, bn=False, upsample=False),
                                        GBlock(1*chn, 2*chn, downsample=True, bn=False, upsample=False),
                                        GBlock(2*chn, 4*chn, downsample=True, bn=False, upsample=False),
                                        GBlock(4*chn, 4*chn, downsample=False, bn=False, upsample=False),
                                        GBlock(4*chn, 8*chn, downsample=True, bn=False, upsample=False),
                                        GBlock(8*chn, 8*chn, downsample=False, bn=False, upsample=False)
                                    ])
        # self.hp = nn.ModuleList([   #HPLayer(4, 2*chn),
        #                             HPLayer(2, 4*chn),
        #                             HPLayer(0, 8*chn)] )

        self.C = combLayer(la=8*chn , lb=self.c_dim, mod=cfgs.c_mod)
        # self.wa = WordAttention()
        self.Cb = combLayer(la=4*chn , lb=self.c_dim, mod="p")
        # first_channel = self.C.out_channels

    def forward(self, images, sentence_emb, words_emb, k=1):
        B = sentence_emb.shape[0]
        out = self.pre_conv(images)
        out = out + self.pre_skip(F.avg_pool2d(images, 2))
        # out = self.s_attn(out)
        # wa = self.wa(words_emb,sentence_emb)
        # h = 0
        # words_emb , mu, logvar = self.CA(words_emb.permute(0,2,1).contiguous().view(-1,768))
        # words_emb = words_emb.view(B,-1,50)

        for i, conv in enumerate(self.encoder):
            """
            if i == 1:
                # out_ga , attn = self.g_attn(out,words_emb)
                # out_sa = self.s_attn(out)
                # out = torch.cat([out_ga, out_sa], 1)
                out = self.s_attn(out)
            """
            if i != 2:
                out = conv(out)
            else:
                skip = conv(out)
                skip = F.avg_pool2d(skip, 2)
                skip = F.relu(skip, True)
                C = skip.shape[1]
                skip = skip.view(B, C, -1)
                skip = skip.sum(2)
            # out = conv(out)

        # loss = []
        # for conv,hp in zip(self.encoder, self.hp):
        #     out = conv(out)
        #     loss.append(hp(out, sentence_emb))

        # out = self.c_attn(out,sentence_emb)
        # out = F.avg_pool2d(out, 2)


        out = F.relu(out, True)
        C = out.shape[1]
        out = out.view(B, C, -1)
        out = out.sum(2)        # ([B, 512])
        # sentence_emb = self.T_linear(sentence_emb)
        # print(words_emb.shape)
        a = self.C(out, words_emb, 1)
        # b = self.Cb(skip, words_emb, 1)

        # a = self.C(out, sentence_emb)
        b = self.Cb(skip, sentence_emb, self.k)
        # b = torch.mean(b,1).unsqueeze(-1)
        return a , b
        # return self.C(out, words_emb)

class Trainer(object):
    """docstring for Network"""
    def __init__(self, cfgs):
        super(Trainer, self).__init__()
        self.z_dim = cfgs.z_dim
        self.chn = cfgs.channels
        self.lambda_gp = 10
        self.cuda_id = cfgs.cuda_id
        
        # self.k_t = 0
        # self.lambda_k = 0.005
        # self.gamma = 0.5
# 
        self.text_encoder = RNN_ENCODER(5450, nhidden=256).cuda(self.cuda_id)
        state_dict = torch.load("../DAMSMencoders/bird/text_encoder200.pth",map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        self.G = Generator(cfgs)
        self.D = Discriminator(cfgs)
        self.G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), cfgs.lr_g, betas=(0.0, 0.9))
        # self.D_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.D.parameters()), cfgs.lr)
        self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), cfgs.lr_d, betas=(0.0, 0.9))

        self.train_step = self.wgan_train_step
        
        self.G.to(cfgs.cuda_id)
        self.D.to(cfgs.cuda_id)
        #self.rnn_encoder.to(self.cuda_id)

        # save path
        self.tb_path = cfgs.tb_path
        self.save_path = cfgs.save_path
        self.batch_size = cfgs.batch_size

        # counters
        self.n_iter = 0
        self.nimgs = 0
        self.img_rounds = 0
        # visualize
        self.normalize_images = cfgs.normalize_images
        self.draw_after_iters = cfgs.draw_after_iters
        #self.de_normlize = None
        self.tensorboard = SummaryWriter(self.tb_path + os.sep + 'log')
        self.to_image = transforms.ToPILImage()
        self.inception_score = False
        self.save_im_count = 0

        # self.criterion = nn.BCELoss()

        x = self.c_pm(self.G)
        y = self.c_pm(self.D)
        print("Gp:%d , Dp:%d" % (x,y))

    def de_normlize(self, img):
        normalized_images = img * 127.5 + 127.5
        return normalized_images

    def learning_rate_decay(self,scale):
        self.G_optimizer.param_groups[0]['lr'] *= scale
        self.D_optimizer.param_groups[0]['lr'] *= scale
        return

    def add_scalars(self,name,value,niter):
        self.tensorboard.add_scalar(name, value, niter)
        return

    def add_images(self, real_pic, fake_pic):
        grid = 5
        nb_images = grid ** 2
        self.nimgs += 1
        if self.nimgs > 10 :
            self.nimgs = 1
            self.img_rounds += 1
        if self.normalize_images:
            real_pic = self.de_normlize(real_pic)
            fake_pic = self.de_normlize(fake_pic)
       
        fake_imgs = make_grid(fake_pic[:nb_images], grid)
        real_imgs = make_grid(real_pic[:nb_images], grid)
        self.tensorboard.add_image('im_s/no.%d_fake' % self.img_rounds ,fake_imgs, self.nimgs)
        self.tensorboard.add_image('im_s/no.%d_real' % self.img_rounds ,real_imgs, self.nimgs)
        return 

    def save_model(self, save_path):
        torch.save(self.G.state_dict(), save_path+ "_iter%d_G.sd" % self.n_iter)
        torch.save(self.D.state_dict(), save_path+ "_iter%d_D.sd" % self.n_iter)
        return
        
    def load_model(self, save_path):
        self.G.load_state_dict(torch.load("%s_G.sd" % save_path))
        self.D.load_state_dict(torch.load("%s_D.sd" % save_path))
        return

    def wgan_train_step(self, batch):
        """
            image {[type]} -- [(B,3,64,64)]
            target {[type]} -- [(B,16,768)]
        """
        # ================== Train D ================== #
        # self.D.train()
        # self.G.train()
        """
        imgs, captions, cap_lens, class_ids, keys = self.prepare_data(batch)
        # image, target, label = batch
        batch_size = captions.shape[0]
        
        hidden = self.text_encoder.init_hidden(batch_size)
        captions, cap_lens = captions.cuda(self.cuda_id), cap_lens.cuda(self.cuda_id)
        words_emb, sent_emb = self.text_encoder(captions, cap_lens, hidden)
        # words_emb, sent_emb = words_emb.cuda(self.cuda_id), sent_emb.cuda(self.cuda_id)
        real_images = imgs[0].cuda(self.cuda_id)
        """

        imgs , captions = batch
        real_images = imgs.cuda(self.cuda_id)
        batch_size = real_images.shape[0]
        words_emb = captions.cuda(self.cuda_id)
        sent_emb = torch.mean(captions,1).cuda(self.cuda_id)
        # print(words_emb.shape,sent_emb.shape)
        
        d_out_real, c_real_labels = self.D(real_images, sent_emb, words_emb, 1)
        # d_loss_real = F.relu(1.0 - (d_out_real+c_real_labels),True).mean()
        d_loss_real = F.relu(1.0 - d_out_real,True).mean() + F.relu(1.0 - c_real_labels,True).mean()

        # apply Gumbel Softmax
        z = torch.randn(batch_size, self.z_dim).to(self.cuda_id)
        
        fake_images = self.G(z, sent_emb, words_emb).detach()
        d_out_fake, c_fake_labels = self.D(fake_images, sent_emb, words_emb, 1)

        # d_loss_fake = d_out_fake.mean()
        # d_loss_fake = F.relu(1.0 + (d_out_fake+c_fake_labels),True).mean()
        d_loss_fake = F.relu(1.0 + d_out_fake,True).mean() + F.relu(1.0 + c_fake_labels,True).mean()
        # d_loss_fake = F.relu(1.0 + (d_out_fake),True).mean()

        # d_loss =  d_loss_real + self.k_t * d_loss_fake
        d_loss =  d_loss_real + d_loss_fake
        self.reset_grad()
        d_loss.backward()
        self.D_optimizer.step()

        # ================== Train G and gumbel ================== #
        # Create random noise
        # if self.n_iter % 5 == 1:
        z = torch.randn(batch_size, self.z_dim).to(self.cuda_id)
        fake_images = self.G(z, sent_emb, words_emb)

        # Compute loss with fake images
        g_out_fake , g_fake_labels= self.D(fake_images, sent_emb, words_emb)  # batch x n
        # g_loss_fake = 5 * self.criterion(g_fake_labels,real_labels) - g_out_fake.mean()
        g_loss_fake =  - (g_out_fake.mean() + g_fake_labels.mean())
            # g_loss_fake =  - (g_fake_labels).mean()

        self.reset_grad()
        g_loss_fake.backward()
        self.G_optimizer.step()

        if self.n_iter % 16 == 8:
            #     self.add_scalars("loss/D_loss_real",d_loss_real.item(),self.n_iter)
            #     self.add_scalars("loss/D_loss_fake",d_loss_fake.item(),self.n_iter)
            #     self.add_scalars("loss/D_loss",d_loss.item(),self.n_iter)
            #     self.add_scalars("loss/G_loss",g_loss_fake.item(),self.n_iter)  

            self.add_scalars("loss/D_loss_real",d_loss_real.item(),self.n_iter)
            self.add_scalars("loss/D_loss_fake",d_loss_fake.item(),self.n_iter)
            self.add_scalars("loss/D_loss",d_loss.item(),self.n_iter)
            self.add_scalars("loss/G_loss",g_loss_fake.item(),self.n_iter)
        if self.n_iter % self.draw_after_iters == 1:
            self.add_images(real_images, fake_images)
            # print(c_real_labels[0],g_fake_labels[0])           
        self.n_iter += 1

    def reset_grad(self):
        self.D_optimizer.zero_grad()
        self.G_optimizer.zero_grad()

    def c_pm(self, model):
        params = list(model.parameters())
        k = 0
        for i in params:
            l = 1
            #print("该层的结构：" + str(list(i.size())))
            for j in i.size():
                l *= j
            #print("该层参数和：" + str(l))
            k = k + l
        return k

    def prepare_data(self,data):
        imgs, captions, captions_lens, class_ids, keys = data

        # sort data by the length in a decreasing order
        sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

        real_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            real_imgs.append((imgs[i]))
            
        captions = captions[sorted_cap_indices].squeeze()
        class_ids = class_ids[sorted_cap_indices].numpy()
        # sent_indices = sent_indices[sorted_cap_indices]
        keys = [keys[i] for i in sorted_cap_indices.numpy()]
        # print('keys', type(keys), keys[-1])  # list
        captions = captions
        sorted_cap_lens = sorted_cap_lens

        return [real_imgs, captions, sorted_cap_lens,class_ids, keys]


    def validate(self,batch):
        # imgs, captions, cap_lens, class_ids, keys = self.prepare_data(batch)
        # # image, target, label = batch
        # batch_size = captions.shape[0]
        
        # hidden = self.text_encoder.init_hidden(batch_size)
        # captions, cap_lens = captions.cuda(self.cuda_id), cap_lens.cuda(self.cuda_id)
        # words_emb, sent_emb = self.text_encoder(captions, cap_lens, hidden)
        # words_emb, sent_emb = words_emb.cuda(self.cuda_id), sent_emb.cuda(self.cuda_id)

        imgs , captions = batch
        real_images = imgs.cuda(self.cuda_id)
        batch_size = real_images.shape[0]
        words_emb = captions.cuda(self.cuda_id)
        sent_emb = torch.mean(captions,1).cuda(self.cuda_id)

        # apply Gumbel Softmax
        z = torch.randn(batch_size, self.z_dim).to(self.cuda_id)
        
        fake_images = self.G(z, sent_emb, words_emb).detach().cpu()


        for i in fake_images:
            self.save_image_id += 1
            image_file_name = "/%d.jpg" % self.save_image_id
            self.to_image(i).save(self.image_path + image_file_name)
        return

    def make_dir(self):
        self.image_path = self.save_path + '/imgs/iter_' + str(self.n_iter)
        os.makedirs(self.image_path)
        self.save_image_id = 0
