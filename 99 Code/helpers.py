import pandas as pd
from copy import deepcopy
from dataset import CustomDatasetStaging
from torchvision.transforms import Compose,Grayscale,Resize,Pad,ToTensor,Normalize,RandomAffine
from torch.nn import Softmax
from torch import argmax
from torchvision.io import read_image
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from tqdm import tqdm
import warnings

def split_train_test(dataset: str, train_size:float = 0.8):
    main_df = pd.read_excel(dataset)
    main_df = main_df.sample(frac=1,random_state = 42)
    train_len = int(len(main_df)*train_size)
    train_set = main_df.iloc[:train_len,:]
    test_set = main_df.iloc[train_len:,:]
    return CustomDatasetStaging(annotations=train_set), CustomDatasetStaging(annotations=test_set)

def image_grid(imgs,rows,cols):
    w,h = imgs[0].size
    grid = Image.new("L",size=(cols*w,rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img,box=(i%cols*w,i//cols*h))
    
    return grid.resize((grid_w*5,grid_h*5))


# def get_transforms_simple(tooth):
#     if tooth == 31:
#         size = (224,224)
#         trans = Compose([
#             Resize(size,antialias=True),
#             Grayscale()
#         ])
#     elif tooth == 33:
#         size = (224,224)
#         trans = Compose([
#             Resize(size,antialias=True),
#             Grayscale(),
#         ])
#     elif tooth == 34:
#         size = (224,224)
#         trans = Compose([
#             Resize(size,antialias=True),
#             Grayscale(),
#         ])
#     elif tooth == 37:
#         size = (224,224)
#         trans = Compose([
#             Resize(size,antialias=True),
#             Grayscale(),
#         ])
#     elif tooth == 38:
#         size = (224,224)
#         trans = Compose([
#             Resize(size,antialias=True),
#             Grayscale(),
#         ])
#     else:
#         size = (0,0)
#         trans = None
#     return size,trans

def clahe_params(tooth):
    params = dict([(31,[1,4]),
                   (33,[3,2]),
                   (34,[1,8]),
                   (37,[2,8]),
                   (38,[1,4])])
    
    return params[tooth]

def get_transforms(size=(224,224),gray=True,randomaffine = True,degrees = 6, translate = (0.1,0.1), scale = (0.9,1.1),fill = 0):
    transforms = []
    if randomaffine:
        transforms.append(RandomAffine(degrees=degrees,translate=translate,scale=scale,fill=fill))
    transforms.append(Grayscale())
    transforms.append(Resize(size,antialias=True))
    return size, Compose(transforms)

def check_duplicates(splits,silent=False):
    """Very slow probably"""
    if not silent:
        print("Checking for duplicates in the fold splits...")  
    duplicates = []
    for fold,(train,test) in enumerate(tqdm(splits,disable=silent)):
        train_ims = []
        test_ims = []       
        for i in range(len(train)):
            if train.iloc[i,0].endswith(".psd"):
                train_im = PSDImage.open(train.iloc[i,0]).composite()
                train_ims.append(train_im.convert("L"))
            elif train.iloc[i,0].endswith(".png"):
                train_ims.append(read_image(train.iloc[i,0])[:3])
            else:
                raise FileNotFoundError("The format is impossible.")
        for j in range(len(test)):
            if test.iloc[j,0].endswith(".psd"):
                test_im = PSDImage.open(test.iloc[j,0]).composite()
                test_ims.append(test_im.convert("L"))
            elif test.iloc[j,0].endswith(".png"):
                test_ims.append(read_image(test.iloc[j,0])[:3])
            else:
                raise FileNotFoundError("The format is impossible.")
        for i in range(len(train)):
            for j in range(len(test)):
                cond = (train_ims[i] == test_ims[j]).all() if train.iloc[0,0].endswith('png') else train_ims[i] == test_ims[j]
                if cond:
                    duplicates.append((train.iloc[i,0],test.iloc[j,0]))
                    warnings.warn(f"Duplicate found in train and test sets, {train.iloc[i,0]} and {test.iloc[j,0]}")
    if not silent:
        print("No duplicate images found.")

def logits_to_probs(logits):
    return(Softmax(dim=0)(logits))

def probs_to_predictions(probs):
    return argmax(probs,dim=1)

def logits_to_predictions(logits):
    return argmax(logits_to_predictions(logits),dim=1)