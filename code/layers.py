# -*- coding:utf-8 -*-
"""
??
"""
#import math
import os
import sys
import json
import random

#_______ torch _______
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init

#_______ models _______
#from BERT.modeling import BertConfig, BertModel
# from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator
# from sg2im.losses import get_gan_losses
# from sg2im.data import utils
# from sg2im import layout


# def upLayer(in_dim, out_dim, spectral_norm=False, transpose=False):
#     """
#     [Upsale the spatial size by a factor of 2]
#     Arguments:
#         in_dim {[type]} -- [description]
#         out_dim {[type]} -- [description]
#     Returns:
#         [layer] -- [description]
#     """
#     if transpose:
#         layers = [  nn.ReLU(True), nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1, bias=False),
#                     nn.BatchNorm2d(out_dim), nn.ReLU(True)  ]
#     else:
#         layers = [nn.ReLU(True), nn.Upsample(scale_factor=2, mode='nearest')]

#     if spectral_norm == True:
#         layers += [     SpectralNorm(nn.Conv2d( in_dim, out_dim, kernel_size=3, stride=1,padding=1)),
#                         nn.ReLU(True),
#                         SpectralNorm(nn.Conv2d( out_dim, out_dim, kernel_size=3, stride=1,padding=1))
#                         ]
#     else:
#         layers += [     nn.Conv2d( in_dim, out_dim, kernel_size=3, stride=1,padding=1),
#                         nn.BatchNorm2d(out_dim),
#                         nn.ReLU(True),
#                         nn.Conv2d( out_dim, out_dim, kernel_size=3, stride=1,padding=1),
#                         nn.BatchNorm2d(out_dim),
#                         ]
#     layer = nn.Sequential(*layers)
#     return layer

# def downLayer(in_dim, out_dim, spectral_norm=False, transpose=False):
#     """
#     [downsampling layer]
#     Arguments:
#         in_dim {[int]} -- [description]
#         out_dim {[int]} -- [description]
#     Returns:
#         [layer] -- [description]
#     """
#     if spectral_norm:
#         layer = nn.Sequential(  
#             SpectralNorm(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)),
#             nn.ReLU(True),
#             SpectralNorm(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)),
#             nn.ReLU(True),
#             nn.AvgPool2d(2)
#             )
#     else:
#         layer = nn.Sequential(  
#             nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(out_dim),    
#             nn.ReLU(True),
#             nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_dim),       
#             nn.ReLU(True),
#             nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_dim)
#             )
#     return layer

# def txtLayer(out_dim,pooling=True):
    
#     if pooling:
#         #s = (in_dim - out_dim) // (out_dim - 1)
#         return nn.AvgPool1d(kernel_size=3, stride=3, padding=0) # 256 wei 
#     else:
#         layers = [  nn.Linear(768, out_dim),
#                     nn.ReLU(True)   ]
#         return nn.Sequential(*layers)


# def refLayer(in_dim, spectral_norm=False):
#     """
#     refine feature map to a 3-dim picture
#     Arguments:
#         in_dim {[int]} -- [feature map]
#     Returns:
#         refLayer [list] -- [picture]
#     """
#     if spectral_norm:
#         layer = nn.Sequential(  SpectralNorm(nn.Conv2d(in_dim, in_dim//2, kernel_size=3, stride=1,padding=1)),
#                                 nn.ReLU(True),
#                                 SpectralNorm(nn.Conv2d(in_dim//2, in_dim//4, kernel_size=3, stride=1,padding=1)),
#                                 nn.ReLU(True),
#                                 SpectralNorm(nn.Conv2d(in_dim//4,3, kernel_size=1, stride=1,padding=0)) )
#     else:
#         layer = nn.Sequential(  nn.Conv2d(in_dim, in_dim//2, kernel_size=3, stride=1,padding=1),
#                                 nn.BatchNorm2d(in_dim//2),
#                                 nn.ReLU(True),
#                                 nn.Conv2d(in_dim//2, in_dim//4, kernel_size=3, stride=1,padding=1),
#                                 nn.BatchNorm2d(in_dim//4),
#                                 nn.ReLU(True),
#                                 nn.Conv2d(in_dim//4,3, kernel_size=1, stride=1,padding=0) )
#     return layer

# class ResBlock(nn.Module):
#     """
#     res Block
#     Extends:
#         nn.Module
#     """
#     def __init__(self, in_dim, out_dim, spectral_norm=True, mod="up"):
#         super(ResBlock, self).__init__()
#         if mod == "up":
#             self.block = upLayer(in_dim, out_dim, spectral_norm=spectral_norm, transpose=False)
#             self.residual = nn.Sequential(  nn.Upsample(scale_factor=2, mode='nearest'),
#                                             nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1,padding=0) )
#         elif mod == "down":
#             self.block = downLayer(in_dim, out_dim, spectral_norm=spectral_norm, transpose=False)
#             self.residual = nn.Sequential(  nn.AvgPool2d(scale_factor=2),
#                                             nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1,padding=0) )

#     def forward(self, x):
#         out = self.block(x) + self.residual(x)
#         return out


# class TextEncoderLayer(object):
#     """
#     text encoder with pretrained BERT model
    
#     [description]
#     """
#     def __init__(self, cuda_id=0):
#         super(TextEncoderLayer, self).__init__()
#         bert_config_file="/home/bupt/bert/uncased_L-12_H-768_A-12/bert_config.json" 
#         init_checkpoint="/home/bupt/bert/uncased_L-12_H-768_A-12/pytorch_model.bin"
#         ## Required parameters
#         self.bert_config_file = bert_config_file
#         self.init_checkpoint = init_checkpoint
#         self.bert_config = BertConfig.from_json_file(self.bert_config_file)
#         self.model = BertModel(self.bert_config)
#         if self.init_checkpoint is not None:
#             self.model.load_state_dict(torch.load(self.init_checkpoint, map_location='cpu'))
#         self.cuda_id = cuda_id
#         self.model.to(torch.device("cuda:0"))
#         return

#     def encode(self,input_ids,input_masks):
#         """
#         [description]
        
#         Arguments:
#             input_ids {[type]} -- [description]
#             input_masks {[type]} -- [description]
        
#         Returns:
#             [type] -- [description]
#         """
#         self.model.eval()
#         all_encoder_layers, x = self.model(input_ids, token_type_ids=None, attention_mask=input_masks)
#         return x

#     def sentence_to_vector(self,ids,masks):
#         vectors = []
#         for input_ids, input_masks in zip(ids,masks):
#             #input_ids , input_masks = input_ids.cuda(self.cuda_id) , input_masks.cuda(self.cuda_id)
#             vectors.append(self.encode(input_ids,input_masks).detach())
#         return torch.stack(vectors)


# class STLayer(nn.Module):
#     """
#     [description]
#     Extends:
#         nn.Module
#     """
#     def __init__(self, H, W=None,cuda_id=0):
#         super(STLayer, self).__init__()
#         self.H = H
#         if W is None:
#             self.W = H
#         else:
#             self.W = W
#         self.cuda_id = cuda_id  

#     def forward(self, vecs, boxes, obj_to_img):
#         """
#         Input:
#             vecs([O, D, H, W]) where D=128 H=W=64
#         Output:
#             sampled(B, D, H, W) where D=768???
#         """
#         grid = layout._boxes_to_grid(boxes, self.H, self.W)
#         grid = grid.cuda(self.cuda_id)
#         sampled = F.grid_sample(vecs, grid) # (B*O, D, H, W)
#         out = layout._pool_samples(sampled, obj_to_img, pooling="sum")
#         return out


# class Embeding_Layer(object):
#     """
#     text encoder with pretrained BERT model
    
#     [description]
#     """
#     def __init__(self, emb_dim, emb_type="label"):
#         super(Embeding_Layer, self).__init__()
#         self.emb_dim = emb_dim
#         if emb_type == "label":
#             self.embeding = nn.Embedding(self.emb_dim, self.emb_dim)
#         elif emb_type == "text":
#             self.embeding = self.text_embeding
#         else:
#             print("error !")

#     def text_embeding(self,inputs):
#         """
#         [description]
        
#         Arguments:
#             inputs {[tensor]} -- [description]
        
#         Returns:
#             [type] -- [description]
#         """
#         B = inputs.shape[0]
#         x = torch.zeros([B, self.emb_dim])
#         return x

# class HPLayer(nn.Module):
#     """docstring for HPLayer"""
#     def __init__(self, scale, chn, lb=768, k=1):
#         super(HPLayer, self).__init__()
#         self.k = k
#         self.chn = chn
#         self.C = combLayer(chn, lb, "p")
#         convs = []
#         convs.append(GBlock(chn, chn, downsample=False, bn=False, upsample=False))
#         if scale > 0 :
#             convs.append(nn.AvgPool2d(scale))
#         convs.append(nn.ReLU(True))
#         self.conv = nn.Sequential(*convs)
#         # self.scale = scale

#     def forward(self, out, sentence_emb):
#         out = self.conv(out)
#         # out = F.relu(out,True)
#         out = out.view(out.shape[0], out.shape[1], -1)
#         out = out.sum(2)
#         a = self.C(out, sentence_emb, self.k)
#         return a

class combLayer(nn.Module):
    def __init__(self, la, lb, mod):
        """
        Arguments:
            a {[type]} -- [(B, 512)]
            b {[type]} -- [(B, 768)]
        """
        super(combLayer, self).__init__()
        self.mod = mod

        if self.mod == "ip":
            #self.L1 = txtLayer(32,False)
            self.L1 = SpectralNorm(nn.Linear(la, 16))
            self.L2 = SpectralNorm(nn.Linear(la, lb))
            self.forward = self.inner_product
            self.out_channels = 16
        elif self.mod == "ct":
            self.forward = self.concat
            self.out_channels = la + lb
        elif self.mod == "p":
            self.L1 = SpectralNorm(nn.Linear(la, lb))
            self.L2 = SpectralNorm(nn.Linear(la, lb))
            self.forward = self.product
            self.out_channels = lb
        elif self.mod == "no":
            self.out_channels = la
            self.forward = self.no_c
        #self.L4x4 = SpectralNorm(nn.Linear(self.out_channels, 8 * 64))

    def inner_product(self, a, b, k=1):
        """
        Arguments:
            a {[type]} -- [(B, 512)]
            b {[type]} -- [(B, 256, 18)]
        Returns:
            [type] -- [description]
        """
        
        B = a.shape[0]
        # a1 = self.L1(a)
        # a2 = self.L2(a).unsqueeze(1) # B,1,768
        # c = torch.bmm(a2,b).squeeze()

        a1 = self.L1(a)
        a2 = self.L2(a).unsqueeze(-1) # B,768,1
        c = torch.bmm(b,a2).squeeze()

        # a2 = self.L2(a).unsqueeze(-1) # B,1,768
        # c = torch.bmm(b,a2).squeeze()
        # print(c.shape,a1.shape)
        return c + a1

    def concat(self, a, b):
        c = torch.cat([a,b],1)
        c = F.relu(self.L4x4(c),True)
        return c

    def product(self, a, b, k=1):
        a1 = self.L1(a)
        a2 = self.L2(a)
        c = k * (a1 * b) + a2
        #c = self.L4x4(c)
        return c

    def no_c(self, a, b):
        #c = self.L4x4(a)
        return a

class ChannelWiseAttention(nn.Module):
    """docstring for ChannelWiseAttention"""
    def __init__(self, chn, c_dim=768):
        super(ChannelWiseAttention, self).__init__()
        self.T_linear = SpectralNorm(nn.Linear(c_dim,chn))

    def forward(self, out, emb):
        cwa = F.leaky_relu(self.T_linear(emb),True)
        out = cwa.unsqueeze(-1).unsqueeze(-1) * out
        return out
        


# _________ from bigGAN _______________

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()

def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

# def leaky_relu(input):
#     return F.leaky_relu(input, negative_slope=0.2)

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation=F.relu):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

        init_conv(self.query_conv)
        init_conv(self.key_conv)
        init_conv(self.value_conv)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = nn.Conv2d(cdf, idf, kernel_size=1, stride=1,padding=0, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn




class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition=788):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)

        self.embed = nn.Linear(n_condition, in_channel* 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        """
        Arguments:
            input {[type]} -- [description]:  size([B, C, n, n])
            class_id {[type]} -- [condition]: size([B , 20+768])
        Returns:
            [type] -- [description]
        """
        out = self.bn(input)
        # print(class_id.dtype)
        # print('class_id', class_id.size()) # torch.Size([4, 148])
        # print(out.size()) #torch.Size([4, 128, 4, 4])
        # class_id = torch.randn(4,1)
        # print(self.embed)
        embed = self.embed(class_id)
        # print('embed', embed.size())
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        # print(beta.size())
        out = gamma * out + beta
        return out

class GBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, n_class=None, bn=True,
                 activation=F.relu, upsample=True, downsample=False, condition_dim=788):
        super().__init__()

        gain = 2 ** 0.5

        self.conv0 = SpectralNorm(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))
        self.conv1 = SpectralNorm(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalNorm(in_channel, condition_dim)
            self.HyperBN_1 = ConditionalNorm(out_channel, condition_dim)

    def forward(self, input, condition=None):
        """
        Arguments:
            input {[type]} -- [description]
        
        Keyword Arguments:
            condition {[type]} -- [B , 20 + 768] (default: {None})
        
        Returns:
            [type] -- [description]
        """
        out = input

        if self.bn:
            # print('condition',condition.size()) #condition torch.Size([4, 148])
            out = self.HyperBN(out, condition)
        out = self.activation(out)

        if self.upsample:
            # TODO different form papers
            out = F.upsample(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # TODO different form papers
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)
        else:
            skip = input
        return out + skip




class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self,cfgs):
        super(CA_NET, self).__init__()
        self.t_dim = 768
        self.c_dim = 50
        self.fc = SpectralNorm(nn.Linear(self.t_dim, self.c_dim * 2, bias=True))
        self.relu = nn.ReLU()
        self.cuda_id = cfgs.cuda_id

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(std.size()).to(self.cuda_id)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class WordAttention(nn.Module):
    """docstring for WordAttention"""
    def __init__(self):
        super(WordAttention, self).__init__()
        self.text_attention = nn.Sequential( SpectralNorm(nn.Linear(16, 16, bias=True)),
                                             nn.Sigmoid() )

    def forward(self, word_emb, sent_emb):
        """
        Arguments:
            word_emb {[[(B, 16, 768)]]} -- [description]
            sent_emb {[[(B, 768)]]} -- [description]
        """
        k = word_emb
        q = sent_emb.unsqueeze(1)
        # print(k.shape,q.shape)
        return self.text_attention(torch.bmm(q,k).squeeze(1))

        