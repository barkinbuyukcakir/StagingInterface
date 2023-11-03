import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datatypes import AttentionDatabase
from transformers import ViTModel, ViTConfig
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from dataset import CustomDatasetStaging
import torchvision.transforms as T

plt.ioff()



class ViT(nn.Module):
    def __init__(self, config = ViTConfig(), num_labels = 10, patch_size = 32) -> None:
        super(ViT,self).__init__()
        self.config = config
        self.config.encoder_stride = patch_size
        self.config.patch_size = patch_size
        self.vit = ViTModel(self.config,add_pooling_layer = False)
        self.dr = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        
        self.attentions = dict()
        self.signifier = f"vit_{patch_size}_stage"
    
    def forward(self,x,gather_attentions:bool = False)  -> torch.Tensor:
        if gather_attentions:
            x = self.vit(x,output_attentions = True)
            attentions = x.attentions
            x = x["last_hidden_state"]
            for i in range(len(attentions)):
                self.attentions[f"encoder_layer_{i}"] = attentions[i].cpu()
        else:
            x = self.vit(x,output_attentions = False)["last_hidden_state"]
        return self.classifier(self.dr(x[:,0,:]))
    
    def __setup_folders(self,tooth,n_splits):
        self.model_dir = f"/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingVITCrossVal/models/{self.signifier}_{tooth}"
        self.attentions_dir = f"/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingVITCrossVal/training_attentions/{self.signifier}_{tooth}"
        os.makedirs(self.model_dir,exist_ok=True)
        os.makedirs(self.attentions_dir,exist_ok=True)
        for i in range(n_splits):
            os.makedirs(os.path.join(self.model_dir,f'fold{i}'),exist_ok=True)
            os.makedirs(os.path.join(self.attentions_dir,f'fold{i}'),exist_ok=True)
            

    def cross_validate(self,tooth,n_splits,transforms):
        """DONT USE I HAVE NO IDEA WHAT THIS DOES"""
        path = f"/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/Attention/StagingData/{tooth}.xlsx"
        df = pd.read_excel(path)
        df = df.sample(frac=1,random_state=42)
        X = KFold(n_splits=n_splits).split(df.index)
        self.splits = []
        for train_ind,test_ind in X:
            self.splits.append((df.loc[train_ind],df.loc[test_ind]))
        
        self.__setup_folders(tooth,n_splits)
        print(f"Saving models and training attentions to:")
        print(self.model_dir)
        print(self.attentions_dir)
        print("--------------------------------------")
        print(f"Train set size per fold: {len(self.splits[0][0])}")
        print(f"Test set size per fold:{len(self.splits[0][1])}")
        print("Starting training...")
        
        transforms = T.Compose(
            T.Grayscale(num_output_channels=1),
            T.Normalize(mean=[0.4002],std=[0.1440])
        )

        for i, (train_ind, test_ind) in enumerate(self.splits):
            train_set = CustomDatasetStaging(df.loc[train_ind],patch_size=self.config.patch_size,transform=transforms)
            test_set = CustomDatasetStaging(df.loc[test_ind],patch_size=self.config.patch_size,transform=transforms)
            ## CREATE INTERMEDIATE ATTENTIONS SET
            ## TRAIN WITH TRAINING ATTENTION SAVING
            ## TEST
            ## SAVE FOLD MODEL BEST AND FINAL
            ## REPEAT


    @torch.no_grad()
    def intermetidate_rollout(self,dataset:Dataset, epoch:int,gpu_id:int,fold:int):
        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{gpu_id}" if use_cuda else 'cpu')
        gather_attentions = True
        att_loader = DataLoader(
            dataset=dataset,
            batch_size = len(dataset),
            shuffle=False
        )
        attentions = None
        for att_batch,att_labels in att_loader:
            output = self.forward(att_batch.to(device),gather_attentions)
            predictions = output.argmax(dim=1)
            attentions = self.attentions
            self.attentions = dict()
        dbase = AttentionDatabase([attentions],predictions.cpu(),att_labels.cpu(),att_batch.cpu())
        ims = []
        masks = []
        for i in range(len(dataset)):
            image,mask,_ = dbase.alt_rollout(i,discard_ratio=0.75,visualize = False)
            ims.append(image)
            masks.append(mask)
        fig,axs = plt.subplots(5,4,sharex=True, sharey=True,figsize=(20,20))
        s = 0
        for j in [0,2]:
            for i in range(5):
                axs[i,j].imshow(ims[i+1+(2*j)].permute(1,2,0).numpy())
                axs[i,j].set_title(f"Stage: {s}")
                axs[i,j+1].imshow(masks[i+1+(2*j)],cmap="jet")
                axs[i,j+1].set_title(f"Mask")
                s+=1
        fig.set_tight_layout("w_pad")
        fig.suptitle("Epoch {}".format(epoch),fontsize = 'xx-large')
        plt.close(fig)
        fig.savefig(f'./training_attentions/{epoch}_vit_16.png')
    
    
    def train(self,dataset:Dataset,epochs:int = 20,learning_rate:float = 1e-4,batch_size:int = 16,**kwargs):
        use_cuda = torch.cuda.is_available()
        gpu_id = kwargs.pop("gpu_id", 0)
        gather_attentions = kwargs.pop('gather_attentions', False)
        attention_set = kwargs.pop("attention_set",None)
        device = torch.device(f"cuda:{gpu_id}" if use_cuda else 'cpu')
        #Load model, loss function and optimizer
        self.to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate, weight_decay=kwargs.pop('weight_decay',0))
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
            for train_batch, train_labels in tqdm(train_loader):
                output = self.forward(train_batch.to(device),gather_attentions)
                loss = loss_fn(output,train_labels.to(device))
                acc = (output.argmax(dim=1) == train_labels.to(device)).sum().item()/batch_size
                train_acc += acc
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                ct+=1
            if attention_set is not None:
                self.intermediate_rollout(attention_set,e,use_cuda,gpu_id,)

            print(f"Epoch {e+1} | Loss: {(train_loss/ct):.3f} | Accuracy: {(train_acc/ct):.3f} ")
    

    def test(self,):
        pass