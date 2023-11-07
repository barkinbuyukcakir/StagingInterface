import os
os.chdir("/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/VisualAttentionProjects/DentalStagingVIT")
import pandas as pd
import numpy as np
from PIL import Image
from datatypes import AttentionDatabase
import pickle
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

'''
We want multiple things:
1 - Confusion matrices
2 - Error per stage distibutions
3 - Scatter plots of label vs pred
'''

urls=["http://127.0.0.1:8050/attention_maps?model_id=1106_130508986229",
     "http://127.0.0.1:8050/attention_maps?model_id=1106_122448955077",
     'http://127.0.0.1:8050/attention_maps?model_id=1029_070732941712',
     'http://127.0.0.1:8050/attention_maps?model_id=1030_051805898357',
     'http://127.0.0.1:8050/attention_maps?model_id=1030_065846916583',
     'http://127.0.0.1:8050/attention_maps?model_id=1030_051805880713',
     'http://127.0.0.1:8050/attention_maps?model_id=1029_064820095140',
     'http://127.0.0.1:8050/attention_maps?model_id=1030_043519342386',
     'http://127.0.0.1:8050/attention_maps?model_id=1030_035652177484',
     'http://127.0.0.1:8050/attention_maps?model_id=1030_060751323615']

ids = [i.split('=')[-1] for i in urls ]


try:
    with open("paper_recs.pkl",'rb') as f:
        records = pickle.load(f)
except FileNotFoundError:
    print("not found.")
    records = {}
    for root,dirs,files in os.walk('./02 Results/test_results'):
        for dir in dirs:
            preds = []
            labels = []
            if f"{dir.split('_')[-2]}_{dir.split('_')[-1]}" in ids: 
                tooth=dir.split("_")[-4]
                patch_size = dir.split("-")[0][-2:]
                dbases = os.listdir(f"{root}/{dir}")    
                for db in dbases:
                    with open(os.path.join(root,dir,db),'rb') as f:
                        dbase = pickle.load(f)
                    preds.append(dbase.predictions)
                    labels.append(dbase.labels)
                records[f"{tooth}_{patch_size}"]={"preds":torch.cat(preds),
                                                "labels":torch.cat(labels)}
    with open("paper_recs.pkl",'wb') as f:
        pickle.dump(records,f)           


for key,val in records.items():
    preds = val["preds"]
    labels = val['labels']
    #Get the confusion matrices:
    cm = confusion_matrix(labels,preds,labels=[i for i in range(10)])
    disp = ConfusionMatrixDisplay(cm,display_labels=[i for i in range(10)])
    disp.plot()
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(f"Tooth {key[:2]} | Patch Size {key[3:]} ")
    fig.savefig(f"./03 Reports/paper_specific/confusion/{key}.png")
    print
    plt.close("all")

#Get the scatter plots 
    fig, ax = plt.subplots(1,1)
    ax.scatter(preds.numpy(),labels.numpy(),c="b")
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Labels")
    ax.set_title (f"Tooth {key[:2]} | Patch Size {key[3:]}")
    fig.savefig(f"./03 Reports/paper_specific/scatters/{key}.png")
    plt.close('all')

#Get the error per stage distributions
    errors_per_stage=[]
    
    for i in range(10):
        error = 0
        if (labels == i).sum() == 0:
            errors_per_stage.append(error)
            continue
        error = ((labels[labels==i] - preds[labels==i]).abs().sum()/sum(labels == i)).item()
        errors_per_stage.append(error)
    
    fig, ax = plt.subplots()
    ax.bar([i for i in range(10)],errors_per_stage,color='b')
    ax.set_xticks([i for i in range(10)])
    ax.set_xlabel("Stages")
    ax.set_ylabel("MAE")
    ax.set_ylim(0,3.2)
    ax.set_title(f"Tooth {key[:2]} | Patch Size {key[3:]} MAE per Stage")
    print
    fig.savefig(f"./03 Reports/paper_specific/errorbars/{key}.png")
    plt.close("all")
        





            
            
