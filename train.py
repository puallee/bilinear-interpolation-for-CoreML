import os
import argparse

from numpy.random import RandomState

from model import *
from utils import *
import matplotlib.pyplot as plt
# python3 train.py --train_path ./data/train_data --workdir ./data/ --model_type mobilenetV2
# class NamespaceUse(argparse):
#     def __setattr__(self,name):
#         print(name)

parser = argparse.ArgumentParser()
#parser.add_argument('--train_path', type=str, default='../data/clip_img')
parser.add_argument('--train_path', type=str, default='./dataset/training')
parser.add_argument('--workdir', type=str, default= './dataset/' )
parser.add_argument('--model_type', default="resnet50", type=str)
#parser.add_argument('--model_type', default="mobilenetV2", type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--max_lr', default=.5, type=float)
parser.add_argument('--loss_window', default=10, type=int)
parser.add_argument('--loss_growth_trsh', default=.5, type=float)
parser.add_argument('--alpha', default=.1, type=float)
parser.add_argument('--wd', default=0., type=float)
parser.add_argument('--freeze_encoder', default=False, type=bool)
parser.add_argument('--max_lr_decay', default=.8, type=float)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--bce_loss_weight', default=.5, type=float)
parser.add_argument('--reduce_lr_patience', default=0, type=int)
parser.add_argument('--reduce_lr_factor', default=0, type=int)
parser.add_argument('--CLR', default=0, type=int)

args = parser.parse_args()
# print(dir(argparse),help(argparse))


#print(os.listdir(args.train_path))
# path_image = list(map(
#     lambda x: x.split('.')[0],
#     filter(lambda x: x.endswith(''), os.listdir(args.train_path))))
# #print(path_image)
# path_images=[]
# for j in range (len(path_image)):
#     sub = list(map(
#         lambda x: x.split('.')[0],
#         filter(lambda x: x.endswith(''), os.listdir(os.path.join('../data/clip_img/'+path_image[j])))))
#     for i in range (len(sub)):
#         imag = list(map(
#         lambda x: x.split('.')[0],
#         filter(lambda x: x.endswith('.jpg'), os.listdir(os.path.join('../data/clip_img/'+path_image[j]+'/'+sub[i])))))


#         #imag=filter(None,imag)
#         imag = [i for i in imag if(len(str(i))!=0)]
#         for k in range (len(imag)):
#             imag[k]=path_image[j]+'/'+sub[i]+'/'+imag[k]
#         path_images=path_images+imag

# print(len(path_images))
path_images = list(map(
    lambda x: x.split('.')[0],
    filter(lambda x: x.endswith('_matte.png'), os.listdir(args.train_path))))
print(path_images[0])

with open('1.txt', 'w') as f:
    for i in range(len(path_images)):
            f.write(path_images[i]+"\n")
prng = RandomState(42)

path_images *= 3
prng.shuffle(path_images)
#train_split = int(len(path_images)*.8)
#train_split = int(len(path_images)*1)
train_images=path_images
val_images = list(map(
    lambda x: x.split('.')[0],
    filter(lambda x: x.endswith('_matte.png'), os.listdir('./dataset/testing'))))
# print(val_images[0])
#train_images, val_images = path_images[:train_split], path_images[train_split:]
#train_images, val_images = path_images[:train_split], path_images[:train_split]
# print(len(train_images))
# print(len(val_images))

dataset = DatasetProcessor(
    args.train_path, train_images, as_torch_tensor=True, augmentations=True, mask_weight=True)
dataset_val = DatasetProcessor(
    './dataset/testing', val_images, as_torch_tensor=True, augmentations=True, mask_weight=True)

model_params = {
    "directory":args.workdir,
    "model":args.model_type,
    "model_name":"%s_model" % (args.model_type),
    "Dropout":.4,
    "device_idx":0,
    "pretrained":True,
    "num_classes":1,
    "num_filters":32,
    "reset":True,
    "ADAM":True
}


trainer = Trainer(**model_params)
trainer.fit(dataset, dataset_val, args)
trainer.plot_trainer_history(mode="loss", save_only=True)
trainer.plot_trainer_history(mode="metric", save_only=True)