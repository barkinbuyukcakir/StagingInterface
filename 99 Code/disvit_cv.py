import os
os.chdir("/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/VisualAttentionProjects/DentalStagingVIT/")
from datetime import datetime
import argparse
import torch.nn as nn
import torchvision.transforms as T
import pandas as pd
from helpers import get_transforms, check_duplicates, clahe_params
from torch.utils.data import DataLoader, Dataset
from torch.cuda import is_available
import torch
from torch.optim import Adam
from dataset import CustomDatasetStaging
from tqdm import tqdm
from datatypes import AttentionDatabase
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score as kappa
from typing import List
import warnings
import numpy as np
from vit_pytorch import ViT,distill
from vit_pytorch.recorder import Recorder     
from torchvision.models import densenet201
warnings.filterwarnings("ignore")
plt.ioff()

class DistillViT(nn.Module):
    def __init__(self,image_size:int | tuple = 224, patch_size: int = 32, num_classes:int = 10 ,depth:int = 6, head_num:int = 8,channels:int =1) -> None:
        super().__init__()
        self.v= distill.DistillableViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = 1024,
            depth = depth,
            heads = head_num,
            mlp_dim = 1024,
            channels=channels,
            dropout=0.1,
            emb_dropout = 0.1
        )

        # self.v = Recorder(self.v)
        t = nn.Sequential(
            densenet201(pretrained = True),
            nn.Linear(1000,10)
        )

        self.distiller = distill.DistillWrapper(
            student=self.v,
            teacher=t,
            temperature=3,
            alpha=0.5,
            hard = False
        )

    def forward(self,x,l):
        return self.distiller(x,l)

class BasicViT(nn.Module):
    def __init__(self,image_size:int | tuple = 224, patch_size: int = 32, num_classes:int = 10 ,depth:int = 12, head_num:int = 16,channels:int =1) -> None:
        super().__init__()
        self.vit = ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = 512,
            depth = depth,
            heads = head_num,
            mlp_dim = 512,
            channels=channels
        )
        self.vit = Recorder(self.vit)
    def forward(self,x):
        return self.vit(x)


class CrossValidator():
    """
    This is the first iteration of the code translated to be modular. Can be optimized with class variables.
    """
    def __init__(self,tooth=31,
                 gpu = 0,
                 epochs = 20,
                 clahe = False,
                 randomaffine = False,
                 patch_size = 32,
                 silent = False,
                 layers = 12,
                 heads = 16,
                 modelType = "BasicVIT") -> None:
   

        self.TOOTH = tooth
        self.GPU= gpu
        epochs = epochs
        clahe = clahe
        ra = randomaffine
        patch_size = patch_size
        self.SILENT = silent
        self.n_layers= layers
        self.n_heads = heads


        dt = datetime.now().strftime("%m%d_%H%M%S%f")

        self.SIGNIFIER = f"{modelType}{patch_size}-{layers}-{heads}_{self.TOOTH}_{'CL' if clahe else ''}{'RA' if ra else ''}_{dt}"


        if not self.SILENT:
            print(f"Cross validation tooth {self.TOOTH} on GPU {self.GPU}")
            print(f"CLAHE: {clahe} | Random Affine Transforms: {ra}")
        path = f"./01 Annotations/{self.TOOTH}.xlsx"

        df = pd.read_excel(path,header=0)
        # mean,std = getMeanStd(df)

        df = df.sample(frac=1,random_state=10)


        try:
            att_set = pd.read_excel(f"./01 Annotations/{self.TOOTH}_att.xlsx")
        except FileNotFoundError:
            att_set = pd.DataFrame(columns = df.columns)
            for i in range(10):
                if len(df[df.Stage == i]) > 0:
                    att_set.loc[i] = df[df.Stage == i].iloc[0]
            # att_set.to_excel(f"./01 Annotations/{self.TOOTH}_att.xlsx",index=False)

        X = KFold(n_splits=5).split(df.index)

        splits = []

        for train_ind,test_ind in X:
            splits.append((df.iloc[train_ind],df.iloc[test_ind]))


        # check_duplicates(splits,silent=self.SILENT)

        size,trans = get_transforms(randomaffine=ra)
        self.cross_validate(
            splits,
            epochs=epochs,
            gpu_id=self.GPU,
            gather_attentions = False,
            batch_size=32,
            learning_rate=1e-5,
            weight_decay= 0.005,
            attention_set= att_set,
            transform = trans,
            image_size = size,
            clahe=clahe,
            randAf=ra,
            patch_size = patch_size
            )
     

    def cross_validate(self,split_datasets: List,epochs:int = 20,learning_rate:float = 1e-4, batch_size:int = 16, weight_decay:float = 0.001,**kwargs):
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
            os.makedirs(f"./02 Results/models/{self.SIGNIFIER}",exist_ok=True)
            os.makedirs(f"./02 Results/test_results/{self.SIGNIFIER}",exist_ok=True)
            cl,ts = clahe_params(self.TOOTH)
            #Only apply RandomAffine for training
            train_set = CustomDatasetStaging(train_df,transform=transform,per_image_norm = False,clahe=clahe,clip_limit = cl, tile_size = (ts,ts),n_channels=3)
            transform = T.Compose([t for t in transform.transforms if not isinstance(t,T.RandomAffine)])
            test_set = CustomDatasetStaging(test_df,transform=transform,per_image_norm = False,clahe=clahe,clip_limit = cl, tile_size = (ts,ts),n_channels=3)
            attention_set = CustomDatasetStaging(attention_df,transform=transform,per_image_norm = False,clahe=clahe,clip_limit = cl, tile_size = (ts,ts),n_channels=3)
            fold_model = self.model_train(
                train_set,
                epochs=epochs,
                gpu_id=self.GPU,
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
            torch.save(fold_model.state_dict(), f"./02 Results/models/{self.SIGNIFIER}/fold{fold}.ckpt")
            fold_model.eval()
            dbase, fold_test_acc,fold_test_loss,fold_kappa = self.model_test(
                test_set,
                trained_model= fold_model,
                gpu_id = self.GPU,
                gather_attentions = True,
                batch_size=1
                )

            with open(f"./02 Results/test_results/{self.SIGNIFIER}/fold{fold}_att.dbase","wb") as f:
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

        print(f"CV complete {self.SIGNIFIER}---> Acc: {cv_acc_mean:.3f} ({cv_acc_std:.3f}) | Kappa: {cv_kappa_mean:.3f} ({cv_kappa_std:.3f})")

    def model_train(self,dataset:Dataset,epochs:int = 20,learning_rate:float = 1e-4,batch_size:int = 16, patch_size:int =32, **kwargs):
        use_cuda = is_available()
        gpu_id = kwargs.pop("gpu_id", 0)
        gather_attentions = kwargs.pop('gather_attentions', False)
        attention_set = kwargs.pop("attention_set",None)
        fold = kwargs.pop("fold",None)
        test_set = kwargs.pop("test_set",None)
        os.makedirs(f"./02 Results/training_attentions/{self.SIGNIFIER}/fold{fold}",exist_ok=True)
        image_size = kwargs.pop("image_size",224)
        device = torch.device(f"cuda:{gpu_id}" if use_cuda else 'cpu')
        #Load model, loss function and optimizer
        # model = ViT(patch_size=patch_size,image_size=image_size).to(device)
        model = DistillViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=10,
            depth=self.n_layers,
            head_num=self.n_heads,
            channels=3
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
            for train_batch, train_labels in tqdm(train_loader,disable=self.SILENT):
                loss = model(train_batch.to(device),train_labels.to(device))
                # loss = loss_fn(output[0],train_labels.to(device))
                # acc = (output[0].argmax(dim=1) == train_labels.to(device)).sum().item()/batch_size
                # train_acc += acc
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                ct+=1
            if attention_set is not None:
                pass
                # logging.getLogger("matplotlib.image").setLevel(logging.ERROR)
                # self.intermediate_rollout(attention_set,e,model,use_cuda,gpu_id,fold,(image_size[1]//patch_size, image_size[0]//patch_size),patch_size)
            if not self.SILENT:
                print(f"FOLD {fold} | Epoch {e+1} | Loss: {(train_loss/ct):.3f} | Accuracy: {(train_acc/ct):.3f} ")
            if test_set is not None:
                    pass
                    # with torch.no_grad():
                    #     _= self.model_test(
                    #         test_set,
                    #         trained_model=model,
                    #         gpu_id = gpu_id,
                    #         gather_attentions=False,
                    #         batch_size=1,
                    #         training=True
                    #     )
        return model

    @torch.no_grad()
    def intermediate_rollout(self,dataset:Dataset, epoch: int, model: str | ViT, use_cuda:bool, gpu_id:int,fold:int,grid_size,patch_size):
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
        dbase = AttentionDatabase([attentions],predictions.cpu(),att_labels.cpu(),att_batch.cpu())
        ims = [None]*10
        masks = [None]*10
        for i in range(len(dataset)):
            image,mask,result,pred,label = dbase.rollout(i,discard_ratio=0.75,visualize = False,grid_size = grid_size)
            ims[label] = image
            masks[label] = mask
        fig,axs = plt.subplots(5,4,sharex=True, sharey=True,figsize=(20,20))
        # for i in range(10):
        #     if ims[i] is None:
        #         ims[i] = torch.ones_like(image)*1000
        #     if masks[i] is None:
        #         masks[i] = torch.ones_like(mask)
            
        s = 0
        for j in [0,2]:
            lol = 5 if j==2 else 0
            for i in range(5):
                if ims[i+lol] is not None:
                    axs[i,j].imshow(ims[i+lol][0].numpy(),cmap="gray")
                    axs[i,j].set_title(f"Stage: {s}")
                    axs[i,j+1].imshow(masks[i+lol],cmap="jet")
                    axs[i,j+1].set_title(f"Mask")
                    s+=1
                else:
                    axs[i,j].imshow(np.ones(image.shape)[0])
                    axs[i,j].add_patch(Rectangle((0,0),mask.shape[0],mask.shape[1],facecolor="white"))
                    axs[i,j].set_title(f"Stage: {s}")
                    axs[i,j+1].imshow(np.ones(image.shape)[0])
                    axs[i,j+1].add_patch(Rectangle((0,0),mask.shape[0],mask.shape[1],facecolor="white"))
                    axs[i,j+1].set_title(f"Mask")
                    s+=1

        fig.set_tight_layout("w_pad")
        fig.suptitle("Epoch {}".format(epoch),fontsize = 'xx-large')
        plt.close(fig)
        fig.savefig(f'./02 Results/training_attentions/{self.SIGNIFIER}/fold{fold}/vit_{epoch:03}.png')

    @torch.no_grad()
    def model_test(self,dataset:Dataset,trained_model:str | ViT,batch_size:int=16,**kwargs):
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
            model = Recorder(trained_model.v.to_vit().to(device))
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
        for test_batch,test_labels in tqdm(test_loader,disable=self.SILENT):
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
        if not self.SILENT:
            print(f"Testing | Loss: {(test_loss/ct):.3f} | Accuracy: {(test_acc/ct):.3f} | Kappa: {k:.3f}\n")
        if not training:
            return (AttentionDatabase(attentions,torch.hstack(predictions),torch.hstack(labels),torch.vstack(images))
        ,test_acc/ct,test_loss/ct,k)

    def load_model(self,path,gpu_id,use_cuda=True):
        device = torch.device(f"cuda:{gpu_id}" if use_cuda else 'cpu')
        #Load model, loss function and optimizer
        model = ViT().to(device)    
        model.load_state_dict(torch.load(path))
        return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tooth",type=int,default=31)
    parser.add_argument("--gpu", type= int, default= 0)
    parser.add_argument("-e","--epochs", type=int,default=20)
    parser.add_argument("--clahe",action="store_true")
    parser.add_argument("-ra","--randomaffine",action="store_true")
    parser.add_argument("-ps","--patch_size",type=int,default=32)
    parser.add_argument("--heads",type=int, default=16)
    parser.add_argument("--layers",type=int,default=12)
    parser.add_argument("--silent",action="store_true")
    args = parser.parse_args()

    CrossValidator(
        tooth=args.tooth,
        gpu = args.gpu,
        epochs=args.epochs,
        clahe=args.clahe,
        randomaffine=args.randomaffine,
        patch_size=args.patch_size,
        silent=args.silent,
        layers = args.layers,
        heads= args.heads
    )