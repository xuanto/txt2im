# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import torch
# import torch.nn as nn
# import json
# from bert_serving.client import BertClient
# from tqdm import trange

import os
import json
import torchvision.transforms as T
from PIL import Image


def generate_word_dict():
    data_kind = "val"
    caproot = "/home/lut/data/COCO2017/annotations/captions_%s2017.json" % data_kind
    with open(caproot,"rb") as f:
        x = json.load(f)
    idx = 1
    word2idx = {}
    idx2word = {}
    x = x["annotations"]
    for txt in x:
        cap = txt["caption"]
        for word in cap.split():
            word = ''.join([i for i in word.lower() if i.isalpha()])
            if word not in word2idx:
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
    with open("/home/lut/data/txt2im/word2idx.json","w") as f:
        json.dump(word2idx,f)
    with open("/home/lut/data/txt2im/idx2word.json","w") as f:
        json.dump(idx2word,f)
    print(len(word2idx),len(idx2word),idx-1)


# def preprocess_data():
#     bc = BertClient()
#     data_kind = "val"
#     caproot = "/home/lut/data/COCO2017/annotations/captions_%s2017.json" % data_kind
#     with open(caproot,"rb") as f:
#         x = json.load(f)
#     nb_caps = len(x['annotations'])
#     for i in trange(nb_caps):
#         cap = x['annotations'][i]["caption"]
#         wd_emb = bc.encode([cap])
#         x['annotations'][i]["caption"] = [cap,wd_emb.tolist()]

#     caproot = "/home/lut/data/COCO2017/annotations/pred_captions_%s2017.json" % data_kind
#     with open(caproot,"w") as f:
#         json.dump(x,f)


# def preprocess_data():
#     caproot = "/home/wangn/data/birds/text/"
#     x = []
#     for _, dirs, _ in os.walk(caproot, topdown=True):
#         for i in dirs:
#             j = os.path.join(caproot, i)
#             for _,_,files in os.walk(j):
#                 for name in files:
#                     y = {}
#                     caption_file_path = os.path.join(j, name)
#                     image_file_path = os.path.join(i, name)[:-4] + ".jpg"
#                     y["file_path"] = image_file_path
#                     with open(caption_file_path,"r") as f:
#                         for line in f.readlines():
#                             if line:
#                                 y["caption"] = line
#                                 x.append(y)
#     print(len(x))
#     with open("/home/wangn/data/birds/cub.json","w") as f:
#         json.dump(x,f)


def preprocess_data():
    caproot = "/home/wangn/data/birds/text/"
    fl = []
    cnt = 0
    ttt = T.ToTensor()
    with open("/home/wangn/data/CUB/birds/CUB_200_2011/bounding_boxes.txt","r") as bf:
        with open("/home/wangn/data/CUB/birds/CUB_200_2011/images.txt","r") as imf:
            for x,y in zip(bf.readlines(),imf.readlines()):
                tmp = {}
                x = x.split()
                y = y.split()
                tmp['index'] = int(x[0])
                tmp['bbox'] = [float(i) for i in x[1:]]
                tmp['filepath'] = y[1]
                image_path = "/home/wangn/data/CUB/birds/CUB_200_2011/images/" + tmp['filepath']
                img = ttt(Image.open(image_path))
                if img.shape[0] != 3:
                    cnt += 1
                    continue
                txt_path = y[1][:-3] + "txt"
                with open("/home/wangn/data/birds/text/"+txt_path,"r") as txf:
                    captions = []
                    for caption in txf.readlines():
                        captions.append(caption)
                tmp['captions'] = captions
                fl.append(tmp)
    print(len(fl))
    print(cnt)
    print(fl[3:5])
    with open("/home/wangn/data/cub_prepaired.json","w") as f:
        json.dump(fl,f)



# def build_vg_dsets(args):
#     with open(args.vocab_json, 'r') as f:
#       vocab = json.load(f)
#     dset_kwargs = {
#       'vocab': vocab,
#       'h5_path': args.train_h5,
#       'image_dir': args.vg_image_dir,
#       'image_size': args.image_size,
#       'max_samples': args.num_train_samples,
#       'max_objects': args.max_objects_per_image,
#       'use_orphaned_objects': args.vg_use_orphaned_objects,
#       'include_relationships': args.include_relationships,
#       'normalize_images':args.normalize_images
#     }
#     train_dset = vg.VgSceneGraphDataset(**dset_kwargs)
#     iter_per_epoch = len(train_dset) // args.batch_size
#     print('There are %d iterations per epoch' % iter_per_epoch)
  
#     dset_kwargs['h5_path'] = args.val_h5
#     del dset_kwargs['max_samples']
#     val_dset = vg.VgSceneGraphDataset(**dset_kwargs)
    
#     return vocab, train_dset, val_dset

# class Coco_collate_fn(object):
#     """docstring for Coco_collate_fn"""
#     def __init__(self):
#         super(Coco_collate_fn, self).__init__()
#         with open("/home/lut/data/txt2im/word2idx.json","rb") as f:
#             self.word2idx = json.load(f)
#             self.max_len = 25

#     def __call__(self, batch):
#         """
#         Collate function to be used when wrapping CocoCaptions in a
#         DataLoader. Returns a tuple of the following:

#         - imgs: FloatTensor of shape (N, C, H, W)
#         - captions: LongTensor of shape (O,) giving object categories
#         """
#         images, captions, captions_lens = [], [], []
#         for i, (img, caps) in enumerate(batch):
#             cp = caps[0].split()
#             if len(cp) > self.max_len:
#                 continue
#             images.append(img)
#             captions_tmp = torch.zeros([self.max_len], dtype=torch.long)
#             length_cnt = 0
#             for j,word in enumerate(cp):
#                 tmp = ''.join([a for a in word.lower() if a.isalpha()])
#                 if tmp not in self.word2idx:
#                     continue
#                 captions_tmp[j] = self.word2idx[tmp]
#                 length_cnt += 1
#             captions_lens.append(length_cnt)
#             captions.append(captions_tmp)
#         #captions = torch.LongTensor(captions)
#         captions_lens = torch.LongTensor(captions_lens) 

#         # sort data by the length in a decreasing order
#         sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
#         #sorted_cap_indices = list(sorted_cap_indices)

#         images = torch.stack(images)
#         captions = torch.stack(captions)

#         images = images[sorted_cap_indices]
#         captions = captions[sorted_cap_indices]

#         return (images,captions,sorted_cap_lens)




if __name__ == '__main__':
    #generate_word_dict()
    #preprocess_data()
    """
    # ！！！ bert 预处理代码 ！！！！
    from train import Cfgs
    from bert_serving.client import BertClient
    from tqdm import trange
    bc = BertClient(check_length=False)
    cfgs = Cfgs()
    host = "/home/lyz/ning"
    data_root = host + "/data"
    # data_path = host + "/data/CUB/birds/CUB_200_2011/images/"
    with open(data_root+"/cub_prepaired.json","rb") as f:
        data_dict = json.load(f)
    print(len(data_dict)) # 11780
    cap_list = []
    for i in trange(len(data_dict)):
        caption = data_dict[i]["captions"][-1]
        cap_list.append(caption)
    captions = bc.encode(cap_list)
    for i in trange(len(data_dict)):
        data_dict[i]["embs"] = captions[i]
    import pickle
    with open(data_root+"/cub_prepaired.pickle","wb") as f:
        pickle.dump(data_dict,f)
    """
    # host = "/home/wangn"
    host = "/home/lyz/ning"
    import pickle
    with open(host+"/data/cub_prepaired.pickle","rb") as f:
        data_dict = pickle.load(f)
    x = []
    label = 1
    cnt = 0
    idx = 0
    for data in data_dict:
        # label = int(data["filepath"][:3])
        # print(data["label"],type(data["label"]))
        if data["label"] == label :
            a,b,c,d = data["bbox"]
            if (c-a) * (d-b) > 256 * 256 :
                data["index"] = idx
                idx += 1
                x.append(data)
                # cnt += 1
                # if cnt == 2:
                #     cnt = 0
                label += 1
        # else:
        #     label = data["label"]
        #     cnt = 0
        #     a,b,c,d = data["bbox"]
        #     if (c-a) * (d-b) > 256 * 256 :
        #         data["index"] = idx
        #         idx += 1
        #         x.append(data)
        #         # cnt += 1
        #         # if cnt == 2:
        #         #     cnt = 0
        #         label += 1

    print(len(x))
    with open(host+"/data/cub_prepaired_sp.pickle","wb") as f:
        pickle.dump(x,f)
"""
    import copy
    from bert_serving.client import BertClient

    bird_list = ["017","182","173","112","143"]
    bert = BertClient()
   
    new_list = []
    for x in data_dict:
        # if x["filepath"][:3] == "175":
        if x["filepath"][:3] in bird_list :
            a,b,c,d = x["bbox"]
            if (c-a) * (d-b) > 256 * 256 :
                for caption in x["captions"]:
                    if len(caption.split()) > 16:
                        continue
                    tmp = copy.copy(x)
                    tmp["captions"] = [caption]
                    tmp["embs"] = bert.encode(tmp["captions"])
                    new_list.append(tmp)
    print(len(new_list))
    with open(host+"/data/cub_5.pickle","wb") as f:
        pickle.dump(new_list,f)

"""



'''


    def wgan_train_step(self, image, target):
        """
            image {[type]} -- [(B,3,64,64)]
            target {[type]} -- [(B,16,768)]
        """
        # ================== Train D ================== #
        # self.D.train()
        # self.G.train()  
        batch_size = image.shape[0]

        real_images = image.cuda(self.cuda_id)
        words_emb = target.cuda(self.cuda_id)
        sent_emb = torch.mean(words_emb,1)
        words_emb = words_emb.squeeze().transpose(1,2)

        d_out_real = self.D(real_images, sent_emb, words_emb)
        # d_loss_real = torch.mean(d_out_real)
        d_loss_real = F.relu(1.0 - d_out_real,True).mean()

        # apply Gumbel Softmax
        z = torch.randn(batch_size, self.z_dim).to(self.cuda_id)
        
        fake_images = self.G(z, sent_emb, words_emb).detach()
        d_out_fake = self.D(fake_images, sent_emb, words_emb)

        # d_loss_fake = d_out_fake.mean()
        d_loss_fake = F.relu(1.0 + d_out_fake,True).mean()

        d_loss =  d_loss_real + d_loss_fake
        # d_loss =  d_loss_real - self.k_t * d_loss_fake
        self.reset_grad()
        d_loss.backward()
        self.D_optimizer.step()

        # Compute gradient penalty
        # alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.cuda_id).expand_as(real_images)
        # interpolated = alpha * real_images + (1 - alpha) * fake_images
        # out = self.D(interpolated, real_labels)
        # grad = torch.autograd.grad(outputs=out,
        #                            inputs=interpolated,
        #                            grad_outputs=torch.ones(out.size()).to(self.cuda_id),
        #                            retain_graph=True,
        #                            create_graph=True,
        #                            only_inputs=True)[0]

        # grad = grad.view(grad.size(0), -1)
        # grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        # d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
        # # Backward + Optimize
        # d_loss = self.lambda_gp * d_loss_gp

        # self.reset_grad()
        # d_loss.backward()
        # self.d_optimizer.step()

        # ================== Train G and gumbel ================== #
        # Create random noise

        z = torch.randn(batch_size, self.z_dim).to(self.cuda_id)
        fake_images = self.G(z, sent_emb, words_emb)

        # Compute loss with fake images
        g_out_fake = self.D(fake_images, sent_emb, words_emb)  # batch x n
        g_loss_fake = - g_out_fake.mean()

        self.reset_grad()
        g_loss_fake.backward()
        self.G_optimizer.step()

        if self.n_iter % self.draw_after_iters == 1:
            self.add_images(real_images, fake_images)
        if self.n_iter % 5 == 4:
            self.add_scalars("loss/D_loss_real",d_loss_real.item(),self.n_iter)
            self.add_scalars("loss/D_loss_fake",d_loss_fake.item(),self.n_iter)
            self.add_scalars("loss/D_loss",d_loss.item(),self.n_iter)
            self.add_scalars("loss/G_loss",g_loss_fake.item(),self.n_iter)       
        self.n_iter += 1





    def began_train_step(self, image, target):   #objs, boxes, crops, obj_to_img):
        """
        Input:
        images      ([B, 3, 64, 64])     objs            ([O])
        boxes       ([O, 4])             crops           ([O, 3, 32, 32])
        obj_to_img  ([O])
        """
        image = image.cuda(self.cuda_id)
        target = target.cuda(self.cuda_id)
        batch_size = image.shape[0]

        z = torch.randn(batch_size, self.z_dim).to(self.cuda_id)
        fake_img = self.G(z,target).detach()
        if self.n_iter % self.draw_after_iters == 1:
            self.add_images(image, fake_img)

        d_real = self.D(image, target)
        d_fake = self.D(fake_img, target)

        # began loss
        # d_loss_real = torch.mean(torch.abs(d_real - image))
        # d_loss_fake = torch.mean(torch.abs(d_fake - fake_img))
        d_loss_real = torch.mean((d_real - image)**2)
        d_loss_fake = torch.mean((d_fake - fake_img)**2)

        D_loss = d_loss_real - self.k_t * d_loss_fake
        # wgan loss
        # D_loss = fake_score.mean() - real_score.mean()
        self.D_optimizer.zero_grad()
        D_loss.backward()
        self.D_optimizer.step()

        z = torch.randn(batch_size, self.z_dim).to(self.cuda_id)
        fake_img = self.G(z,target)
        d_fake = self.D(fake_img, target)
        # began loss
        # G_loss = torch.mean(torch.abs(d_fake - fake_img))
        G_loss = torch.mean((d_fake - fake_img)**2)

        # wgan loss
        # G_loss = -fake_score
        self.G_optimizer.zero_grad()
        G_loss.backward()
        self.G_optimizer.step()

        balance = (self.gamma * d_loss_real - G_loss).item()
        measure = d_loss_real.item() + abs(balance)
        self.k_t += self.lambda_k * balance
        self.k_t = max(min(1, self.k_t), 0)
        
        if self.n_iter % 5 == 4:
            self.add_scalars("loss/D_loss",D_loss.item(),self.n_iter)
            self.add_scalars("loss/G_loss",G_loss.item(),self.n_iter)
            self.add_scalars("hp/balance",balance,self.n_iter)
            self.add_scalars("hp/measure",measure,self.n_iter)
            self.add_scalars("hp/k_t",self.k_t,self.n_iter)
        self.n_iter += 1
        return



'''