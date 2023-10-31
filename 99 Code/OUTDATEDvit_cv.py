import os
import sys
from contextlib import contextmanager
from datetime import datetime
os.chdir("/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingVITCrossVal/")
import argparse
import torch.nn as nn
import torchvision.transforms as T
import pandas as pd
from helpers import split_train_test, get_transforms, image_grid, check_duplicates, clahe_params
from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import DataLoader, Dataset
from transformers import ViTModel, ViTConfig
from torch.cuda import is_available
import torch
from torch.optim import Adam, AdamW, RMSprop
from dataset import CustomDatasetStaging
from tqdm import tqdm
from datatypes import AttentionDatabase
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score as kappa
from typing import List
import warnings
import logging
import time
from vit_pytorch import SimpleViT,ViT
from vit_pytorch.recorder import Recorder
warnings.filterwarnings("ignore")
plt.ioff()



# DEPRECATED!!
# class ViTold(nn.Module):
#     def __init__(self, config = ViTConfig(), num_labels = 10, model_ckpt = "google/vit-base-patch16-224-in21k",patch_size:int = 32,image_size = (352,128)) -> None:
#         super(ViT,self).__init__()
#         config.encoder_stride = patch_size
#         config.patch_size = patch_size
#         config.num_channels = 1
#         # config.image_size = image_size                                  
#         self.vit = ViTModel(config,add_pooling_layer = False)
#         self.classifier = nn.Linear(config.hidden_size,num_labels)
#         self.dr =  nn.Dropout(p=0.1)
#         self.attentions = dict()
#     def forward(self,x,gather_attentions:bool = False)  -> torch.Tensor:
#         if gather_attentions:
#             x = self.vit(x,output_attentions = True)
#             attentions = x.attentions
#             x = x["last_hidden_state"]
#             for i in range(len(attentions)):
#                 self.attentions[f"encoder_layer_{i}"] = attentions[i].cpu()
#         else:
#             x = self.vit(x,output_attentions = False)
#             x = x["last_hidden_state"]
#         return self.classifier(x[:,0,:])

class NewViT(nn.Module):
    def __init__(self,image_size:int | tuple = 224, patch_size: int = 32, num_classes:int = 10 ,depth:int = 6, head_num:int = 16,channels:int =1) -> None:
        super().__init__()
        self.vit = ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = 1024,
            depth = depth,
            heads = head_num,
            mlp_dim = 2048,
            channels=channels
        )
        self.vit = Recorder(self.vit)
    def forward(self,x):
        return self.vit(x)
    

        

def cross_validate(split_datasets: List,epochs:int = 20,learning_rate:float = 1e-4, batch_size:int = 16, weight_decay:float = 0.001,**kwargs):
    #TODO: Build wrapper to AttentionDatabase (or sth. idk) to calculate and track metrics instead of calculating in train and test loops.
    cv_accs = []
    cv_losses= []
    cv_kappas = []
    use_cuda = is_available()
    gpu_id = kwargs.pop("gpu_id", 0)
    gather_attentions = kwargs.pop('gather_attentions', False)
    attention_df = kwargs.pop("attention_set",None)
    patch_size = kwargs.pop("patch_size",32)
    image_size = kwargs.pop("image_size",(224,224))
    transform = kwargs.pop("transform",None)
    clahe=kwargs.pop("clahe",False)
    ra = kwargs.pop("randAf",False)
    
    for fo,(train_df,test_df) in enumerate(split_datasets):
        fold = fo+1
        os.makedirs(f"./models/{SIGNIFIER}",exist_ok=True)
        os.makedirs(f"./test_results/{SIGNIFIER}",exist_ok=True)
        cl,ts = clahe_params(TOOTH)
        train_set = CustomDatasetStaging(train_df,transform=transform,per_image_norm = False,clahe=clahe,clip_limit = cl, tile_size = (ts,ts))
        test_set = CustomDatasetStaging(test_df,transform=transform,per_image_norm = False,clahe=clahe,clip_limit = cl, tile_size = (ts,ts))
        attention_set = CustomDatasetStaging(attention_df,transform=transform,per_image_norm = False,clahe=clahe,clip_limit = cl, tile_size = (ts,ts))
        fold_model = model_train(
            train_set,
            epochs=epochs,
            gpu_id=GPU,
            gather_attentions = gather_attentions,
            batch_size=batch_size,
            patch_size=patch_size,
            learning_rate=learning_rate,
            weight_decay= weight_decay,
            attention_set= attention_set,
            fold = fold,
            image_size = image_size,
            test_set = test_set,
            )
        torch.save(fold_model.state_dict(), f"./models/{SIGNIFIER}/fold{fold}.ckpt")
        fold_model.eval()
        dbase, fold_test_acc,fold_test_loss,fold_kappa = model_test(
            test_set,
            trained_model= fold_model,
            gpu_id = GPU,
            gather_attentions = True,
            batch_size=1
            )
        with open(f"./test_results/{SIGNIFIER}/fold{fold}_att.dbase","wb") as f:
            pickle.dump(dbase,f)

        cv_accs.append(fold_test_acc)
        cv_losses.append(fold_test_loss)
        cv_kappas.append(fold_kappa)

    #TODO: Clean this up
    cv_accs = torch.Tensor(cv_accs)
    cv_losses = torch.Tensor(cv_losses)
    cv_kappas = torch.Tensor(cv_kappas)

    cv_loss_mean = cv_losses.mean()
    cv_loss_std = cv_losses.std()
    cv_acc_mean = cv_accs.mean()
    cv_acc_std = cv_accs.std()
    cv_kappa_mean = cv_kappas.mean()
    cv_kappa_std = cv_kappas.std()

    print(f"CV complete {SIGNIFIER}---> Acc: {cv_acc_mean:.3f} ({cv_acc_std:.3f}) | Kappa: {cv_kappa_mean:.3f} ({cv_kappa_std:.3f})")

def model_train(dataset:Dataset,epochs:int = 20,learning_rate:float = 1e-4,batch_size:int = 16, patch_size:int =32, **kwargs):
    use_cuda = is_available()
    gpu_id = kwargs.pop("gpu_id", 0)
    gather_attentions = kwargs.pop('gather_attentions', False)
    attention_set = kwargs.pop("attention_set",None)
    fold = kwargs.pop("fold",None)
    test_set = kwargs.pop("test_set",None)
    os.makedirs(f"./training_attentions/{SIGNIFIER}/fold{fold}",exist_ok=True)
    image_size = kwargs.pop("image_size",224)
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else 'cpu')
    #Load model, loss function and optimizer
    # model = ViT(patch_size=patch_size,image_size=image_size).to(device)
    model = NewViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=10,
        depth=12,
        head_num=16,
        channels=1
    ).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(),lr=learning_rate, weight_decay=kwargs.pop('weight_decay',0))
    #Load batch image
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    for e in range(epochs):
        train_acc = 0
        train_loss = 0
        ct = 0
        for train_batch, train_labels in tqdm(train_loader,disable=SILENT):
            output = model(train_batch.to(device))
            loss = loss_fn(output[0],train_labels.to(device))
            acc = (output[0].argmax(dim=1) == train_labels.to(device)).sum().item()/batch_size
            train_acc += acc
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            ct+=1
        if att_set is not None:
            pass
            logging.getLogger("matplotlib.image").setLevel(logging.ERROR)
            intermediate_rollout(attention_set,e,model,use_cuda,gpu_id,fold,(image_size[1]//patch_size, image_size[0]//patch_size),patch_size)
        if not SILENT:
            print(f"FOLD {fold} | Epoch {e+1} | Loss: {(train_loss/ct):.3f} | Accuracy: {(train_acc/ct):.3f} ")
        if test_set is not None:
                pass
                # with torch.no_grad():
                #     _= model_test(
                #         test_set,
                #         trained_model=model,
                #         gpu_id = gpu_id,
                #         gather_attentions=False,
                #         batch_size=1,
                #         training=True
                #     )
    return model

@torch.no_grad()
def intermediate_rollout(dataset:Dataset, epoch: int, model: str | ViT, use_cuda:bool, gpu_id:int,fold:int,grid_size,patch_size):
    use_cuda = is_available()
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else 'cpu')
    gather_attentions = True
    att_loader = DataLoader(
        dataset=dataset,
        batch_size = len(dataset),
        shuffle=False
    )
    attentions = None
    for att_batch,att_labels in att_loader:
        output = model(att_batch.to(device))
        predictions = output[0].argmax(dim=1)
        attentions = output[1]
        model.attentions = dict()
    dbase = AttentionDatabase([attentions],predictions.cpu(),att_labels.cpu(),att_batch.cpu())
    ims = []
    masks = []
    for i in range(len(dataset)):
        image,mask,result,pred,label = dbase.rollout(i,discard_ratio=0.75,visualize = False,grid_size = grid_size)
        ims.append(image)
        masks.append(mask)
    fig,axs = plt.subplots(5,4,sharex=True, sharey=True,figsize=(20,20))
    s = 0
    for j in [0,2]:
        for i in range(5):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    axs[i,j].imshow(ims[i+1+(2*j)].permute(1,2,0).numpy(),cmap="gray")
                    axs[i,j].set_title(f"Stage: {s}")
                    axs[i,j+1].imshow(masks[i+1+(2*j)],cmap="jet")
                    axs[i,j+1].set_title(f"Mask")
                    s+=1

            except:
                axs[i,j].imshow(torch.ones_like(ims[0]).permute(1,2,0).numpy(),cmap="gray")
                axs[i,j].set_title(f"Stage: {s}")
                axs[i,j+1].imshow(torch.ones_like(torch.Tensor(masks[0])).numpy(),cmap="jet")
                axs[i,j+1].set_title(f"Mask")
                s+=1
    fig.set_tight_layout("w_pad")
    fig.suptitle("Epoch {}".format(epoch),fontsize = 'xx-large')
    plt.close(fig)
    fig.savefig(f'./training_attentions/{SIGNIFIER}/fold{fold}/vit_{epoch:03}.png')

@torch.no_grad()
def model_test(dataset:Dataset,trained_model:str | ViT,batch_size:int=16,**kwargs):
    use_cuda = is_available()
    gpu_id = kwargs.pop("gpu_id", 0)
    gather_attentions = kwargs.pop('gather_attentions', True)
    training = kwargs.pop("training", False)
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else 'cpu')
    #Load the trained model state_dict
    if isinstance(trained_model,str):
        model = ViT().to(device)
        st_dict = torch.load(trained_model)
        model.load_state_dict(state_dict=st_dict)
    else:
        model = trained_model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    test_loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_acc = 0
    test_loss = 0 
    test_kappa = 0
    ct = 0
    images = []
    predictions = []
    labels = []
    attentions = []
    att,im = None,None
    att_old = None
    for test_batch,test_labels in tqdm(test_loader,disable=SILENT):
        output = model(test_batch.to(device))
        images.append(test_batch.cpu())
        attentions.append(output[1])
        loss = loss_fn(output[0],test_labels.to(device))
        preds = output[0].argmax(dim=1)
        acc = (preds == test_labels.to(device)).sum().item()/batch_size
        test_acc += acc
        test_loss += loss.item()
        predictions.append(preds.cpu())
        labels.append(test_labels.cpu())
        ct+=1
    k = kappa(torch.hstack(predictions).numpy(),torch.hstack(labels).numpy())
    if not SILENT:
        print(f"Testing | Loss: {(test_loss/ct):.3f} | Accuracy: {(test_acc/ct):.3f} | Kappa: {k:.3f}\n")
    if not training:
        return (AttentionDatabase(attentions,torch.hstack(predictions),torch.hstack(labels),torch.vstack(images))
    ,test_acc/ct,test_loss/ct,k)

def load_model(path,gpu_id,use_cuda=True):
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else 'cpu')
    #Load model, loss function and optimizer
    model = ViT().to(device)    
    model.load_state_dict(torch.load(path))
    return model




parser = argparse.ArgumentParser()
parser.add_argument("--tooth",type=int,default=31)
parser.add_argument("--gpu", type= int, default= 0)
parser.add_argument("-e","--epochs", type=int)
parser.add_argument("--clahe",action="store_true")
parser.add_argument("-ra","--randomaffine",action="store_true")
parser.add_argument("-ps","--patch_size",type=int,default=32)
parser.add_argument("--silent",action="store_true")
args = parser.parse_args()

TOOTH = int(args.tooth)
GPU= int(args.gpu)
epochs = args.epochs
clahe = args.clahe
ra = args.randomaffine
patch_size = args.patch_size
SILENT = args.silent


dt = datetime.now().strftime("%m%d_%H%M%S%f")

SIGNIFIER = f"VIT{patch_size}_{TOOTH}_{'CL' if clahe else ''}{'RA' if ra else ''}_{dt}"

# def getMeanStd(df):
#     dataset = CustomDatasetStaging(df,transform=get_transforms_simple(tooth=TOOTH)[1])
#     loader = DataLoader(dataset,
#                             batch_size=10,
#                             num_workers=0,
#                             shuffle=False)

#     mean = 0.
#     std = 0.
#     for images, _ in loader:
#         batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
#         images = images.view(batch_samples, images.size(1), -1)
#         mean += images.mean(2).sum(0)
#         std += images.std(2).sum(0)

#     mean /= len(loader.dataset)
#     std /= len(loader.dataset)
#     return mean,std
if not SILENT:
    print(f"Cross validation tooth {TOOTH} on GPU {GPU}")
    print(f"CLAHE: {clahe} | Random Affine Transforms: {ra}")
path = f"/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingData/{TOOTH}.xlsx"

df = pd.read_excel(path,header=0)
# mean,std = getMeanStd(df)

df = df.sample(frac=1,random_state=10)


try:
    att_set = pd.read_excel(f"/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingData/{TOOTH}_att.xlsx")
except FileNotFoundError:
    att_set = pd.DataFrame(columns = df.columns)
    for i in range(10):
        if len(df[df.Stage == i]) > 0:
            att_set.loc[i] = df[df.Stage == i].iloc[0]
    att_set.to_excel(f"/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingData/{TOOTH}_att.xlsx",index=False)

X = KFold(n_splits=5).split(df.index)

splits = []

for train_ind,test_ind in X:
    splits.append((df.iloc[train_ind],df.iloc[test_ind]))


check_duplicates(splits,silent=SILENT)

size,trans = get_transforms(randomaffine=ra)
cross_validate(
    splits,
    epochs=epochs,
    gpu_id=GPU,
    gather_attentions = False,
    batch_size=32,
    learning_rate=1e-5,
    weight_decay= 0.0005,
    attention_set= att_set,
    transform = trans,
    image_size = size,
    clahe=clahe,
    randAf=ra
    )
