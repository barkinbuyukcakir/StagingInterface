"""
This will do multiprocessing 5-fold cross-validation on 5 gpus simultaneously.
"""


import pandas as pd
import torch
import pickle
import os
import argparse
from dataset import CustomDatasetStaging
from tqdm import tqdm
from PIL import Image
import numpy as np
from helpers import get_transforms
import multiprocessing as mp
from autoencoders.models import ConvAutoencoder,LatentClassifier
import torchvision.transforms as T
import matplotlib.pyplot as plt

def fold_train(dataset,device,epochs,embedded_dim = 10,out_range = 1):
    autoencoder = ConvAutoencoder(3,embed_dim=embedded_dim,out_range=out_range).to(device)
    classifier = LatentClassifier(10,embed_dim =embedded_dim).to(device)
    loss_fn_ae = torch.nn.MSELoss().to(device)
    loss_fn_cls = torch.nn.CrossEntropyLoss().to(device)
    optim_ae = torch.optim.Adam(autoencoder.parameters())
    optim_cls = torch.optim.Adam(classifier.parameters())
    loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 32,
        shuffle = True
    )
    pbar = tqdm(range(epochs))
    fig,axs  = plt.subplots(1,2)

    for e in pbar:
        epoch_loss_ae = 0
        epoch_loss_cls = 0 
        for train_batch,train_label in loader:
            optim_ae.zero_grad()
            optim_cls.zero_grad()
            out,embeddings = autoencoder(train_batch.to(device))
            preds = classifier(embeddings)
            loss_ae = loss_fn_ae(out,train_batch.to(device))
            loss_cls = loss_fn_cls(embeddings,train_label.to(device))
            loss = loss_ae + loss_cls
            epoch_loss_ae += loss_ae.item()/len(loader)
            epoch_loss_cls += loss_cls.item()/len(loader)

            loss.backward()
            optim_ae.step()
            optim_cls.step()
        pbar.set_description(f"{epoch_loss_ae:.3f} | {epoch_loss_cls:.3f}")
        evaluate(autoencoder,dataset,torch.randint(low= 0, high= len(dataset),size= (1,)).item(),device,axs)
    return autoencoder,classifier

@torch.no_grad()
def fold_test(autoencoder,classifier,dataset,device):
    loss_fn_ae = torch.nn.MSELoss().to(device)
    loss_fn_cls = torch.nn.CrossEntropyLoss().to(device)
    loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 32,
        shuffle = True
    )
    loss_ae = 0
    loss_cls = 0
    fig,axs  = plt.subplots(1,2)
    for train_batch,train_label in loader:
        out,embeddings = autoencoder(train_batch.to(device))
        preds = classifier(embeddings)
        loss_ae += loss_fn_ae(out,train_batch.to(device))/len(loader)
        loss_cls += loss_fn_cls(embeddings,train_label.to(device))/len(loader)
        
    evaluate(autoencoder,dataset,0,device,axs)
    print(f"AE Loss {loss_ae:.3f} CLS Loss {loss_cls:.3f}")


    

@torch.no_grad()
def evaluate(model,dataset,index,device,axs = None):
    im = dataset[index][0].unsqueeze(0)
    out,_ = model(im.to(device))
    if axs is None:
        fig,axs  = plt.subplots(1,2)
    else:
        axs[0].imshow(im[0].permute(1,2,0).numpy())
        axs[1].imshow(out[0].permute(1,2,0).cpu().numpy())
    plt.pause(0.2)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--tooth",type=int)
    parser.add_argument("-e","--epochs", type=int,default=20)
    parser.add_argument("--silent",action="store_true")
    args = parser.parse_args()

    TOOTH = args.tooth
    EPOCHS = args.epochs
    SILENT = args.silent

    path = f"./01 Annotations/{TOOTH}.xlsx"
    df = pd.read_excel(path,header=0)
    df = df.sample(frac=1,random_state=42)
    size, trans = get_transforms(randomaffine=False)
    
    tf= int(len(df)*0.8)
    train_set = CustomDatasetStaging(df.iloc[:tf],transform=trans,per_image_norm = True,clahe=False,n_channels=3,range="1")
    test_set = CustomDatasetStaging(df[tf:],transform=trans,per_image_norm = True,clahe=False,n_channels=3,range="1")
    
    trained_autoencoder,trained_classifier = fold_train(train_set,torch.device(f"cuda:0"),EPOCHS)
    fold_test(trained_autoencoder,trained_classifier,test_set,torch.device("cuda:0"))
    print()


    