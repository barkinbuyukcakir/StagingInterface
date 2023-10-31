import os
os.chdir("/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingVITCrossVal/")
import argparse
import torch.nn as nn
import torchvision.transforms as T
import pandas as pd
from helpers import split_train_test, get_transforms, image_grid, get_transforms_simple
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
from typing import List
import warnings
import logging
from vit_pytorch import SimpleViT,ViT
from vit_pytorch.recorder import Recorder
warnings.filterwarnings("ignore")
plt.ioff()



#Deprecated
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
    cv_acc = 0
    cv_loss = 0
    use_cuda = is_available()
    gpu_id = kwargs.pop("gpu_id", 0)
    gather_attentions = kwargs.pop('gather_attentions', False)
    attention_df = kwargs.pop("attention_set",None)
    patch_size = kwargs.pop("patch_size",32)
    image_size = kwargs.pop("image_size",(224,224))
    transform = kwargs.pop("transform",None)
    clahe=kwargs.pop("clahe",False)
    improvement=""
    if clahe:
        improvement+="_clahe"
    clip_limits = [1,2,3,4,5,6,7,8]
    tile_sizes = [2,4,8,16]
    for cl in clip_limits:
        for ts in tile_sizes:
            for fo,(train_df,test_df) in enumerate(split_datasets[:1]):
                fold = fo+1
                os.makedirs(f"./models/vit{patch_size}_stage_{TOOTH}{improvement}",exist_ok=True)
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
                    improvement=improvement
                    )
                torch.save(fold_model.state_dict(), f"./models/vit{patch_size}_stage_{TOOTH}{improvement}/fold{fold}.ckpt")
                fold_model.eval()
                fold_test_acc,fold_test_loss = model_test(
                    test_set,
                    trained_model= fold_model,
                    gpu_id = GPU,
                    gather_attentions = False,
                    batch_size=1
                    )

                cv_acc+=fold_test_acc
                cv_loss+=fold_test_loss
                print(f"Clip Limit: {cl} | Tile Size: {ts} | Test_acc: {fold_test_acc} ")
    # print(f"CV complete ---> Loss {cv_loss/len(split_datasets):.3f} | Acc: {cv_acc/len(split_datasets):.3f}")

def model_train(dataset:Dataset,epochs:int = 20,learning_rate:float = 1e-4,batch_size:int = 16, patch_size:int =32, **kwargs):
    use_cuda = is_available()
    gpu_id = kwargs.pop("gpu_id", 0)
    gather_attentions = kwargs.pop('gather_attentions', False)
    attention_set = kwargs.pop("attention_set",None)
    fold = kwargs.pop("fold",None)
    test_set = kwargs.pop("test_set",None)
    improvement = kwargs.pop("improvement","")
    os.makedirs(f"./training_attentions/vit{patch_size}_stage_{TOOTH}{improvement}/fold{fold}",exist_ok=True)
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
    eps = tqdm(range(epochs))
    for e in eps:
        train_acc = 0
        train_loss = 0
        ct = 0
        for train_batch, train_labels in train_loader:
            output = model(train_batch.to(device))
            loss = loss_fn(output[0],train_labels.to(device))
            acc = (output[0].argmax(dim=1) == train_labels.to(device)).sum().item()/batch_size
            train_acc += acc
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            ct+=1
            eps.set_description(f"{acc:.3f}")
        if att_set is not None:
            pass
            # logging.getLogger("matplotlib.image").setLevel(logging.ERROR)
            # intermediate_rollout(attention_set,e,model,use_cuda,gpu_id,fold,(image_size[1]//patch_size, image_size[0]//patch_size),patch_size,improvement=improvement)

        # print(f"FOLD {fold} | Epoch {e+1} | Loss: {(train_loss/ct):.3f} | Accuracy: {(train_acc/ct):.3f} ")
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
def intermediate_rollout(dataset:Dataset, epoch: int, model: str | ViT, use_cuda:bool, gpu_id:int,fold:int,grid_size,patch_size,improvement):
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
        image,mask,result,pred,label = dbase.alt_rollout(i,discard_ratio=0.75,visualize = False,grid_size = grid_size)
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
    fig.savefig(f'./training_attentions/vit{patch_size}_stage_{TOOTH}{improvement}/fold{fold}/vit_{epoch:03}.png')

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
    ct = 0
    images = []
    predictions = []
    labels = []
    attentions = []
    att,im = None,None
    att_old = None
    for test_batch,test_labels in test_loader:
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
    print(f"Testing | Loss: {(test_loss/ct):.3f} | Accuracy: {(test_acc/ct):.3f} \n")
    if not training:
        return (test_acc/ct,test_loss/ct)

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
args = parser.parse_args()

TOOTH = int(args.tooth)
GPU= int(args.gpu)
epochs = args.epochs
clahe = args.clahe


def getMeanStd(df):
    dataset = CustomDatasetStaging(df,transform=get_transforms_simple(tooth=TOOTH)[1])
    loader = DataLoader(dataset,
                            batch_size=10,
                            num_workers=0,
                            shuffle=False)

    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean,std

print(f"Cross validation tooth {TOOTH} on GPU {GPU}")

path = f"/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingData/{TOOTH}.xlsx"

df = pd.read_excel(path)
# mean,std = getMeanStd(df)

df = df.sample(frac=1,random_state=10)


try:
    att_set = pd.read_excel(f"/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingData/{TOOTH}_att.xlsx")
except FileNotFoundError:
    att_set = pd.DataFrame(columns = df.columns)
    for i in range(10):
        if len(df[df.Stage == i]) > 0:
            att_set.loc[i] = df[df.Stage == i].iloc[0]
    # att_set.to_excel(f"/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingData/{TOOTH}_att.xlsx",index=False)

X = KFold(n_splits=5).split(df.index)

splits = []

for train_ind,test_ind in X:
    splits.append((df.iloc[train_ind],df.iloc[test_ind]))

print(clahe)
cross_validate(
    splits,
    epochs=epochs,
    gpu_id=GPU,
    gather_attentions = False,
    batch_size=32,
    learning_rate=1e-5,
    weight_decay= 0.0005,
    attention_set= att_set,
    transform = get_transforms_simple(TOOTH)[1],
    image_size = get_transforms_simple(TOOTH)[0],
    clahe=clahe
    )

print()