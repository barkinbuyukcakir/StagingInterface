import os
os.chdir("/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/VisualAttentionProjects/DentalStagingVIT")
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#Can Add accuracy cutoff here: only include the model if it's above a certain accuracy
mean_attentions =dict([(str(i),[]) for i in range(10)])
wmean_attentions = dict([(str(i),[]) for i in range(10)])
for root,dirs,files in os.walk('./03 Reports/mean_attention_maps'):
    for name in files:
        if root.__contains__("VIT16"):
            stage = name.split("_")[1]
            if name.split("_")[-1].startswith("wMean"):
                wmean_attentions[stage] += [os.path.join(root,name)]
            elif name.split("_")[-1].startswith("atMean"):
                mean_attentions[stage] += [os.path.join(root,name)]


data = {}
c = 0
n_components=4
pca = PCA(n_components=n_components)
for i in range(10):
    files = mean_attentions[str(i)]
    for f in files:
        im = np.array(Image.open(f).convert("L").resize((224,224)))/255
        data[str(c)] = np.append(im.flatten(),int(f.split("/")[-1].split("_")[1]))
        c+=1
df = pd.DataFrame.from_dict(data).transpose()
df.loc[:,df.columns[-1]]=df.iloc[:,-1].astype(np.uint8)
features = df.iloc[:,:-1]
# features = StandardScaler().fit_transform(features.values)
labels = df.iloc[:,-1]
pca_ims=pca.fit_transform(features)
pc_values = pd.DataFrame(data = pca_ims,columns=[f'PC{i}' for i in range(1,n_components+1)])


plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title(f"Total Explained Variance {sum(pca.explained_variance_ratio_[:2]):.3f}%")
targets = [i for i in range(10)]
for target in targets:
    indicesToKeep = (labels == target).values
    if target<4:
        c = "r"
    elif 4<=target<7:
        c = "g"
    else:
        c = "b"
    plt.scatter(pc_values.loc[indicesToKeep, 'PC1']
               , pc_values.loc[indicesToKeep, 'PC2'], s = 40)

plt.legend(targets,prop={'size': 10})
plt.savefig("./03 Reports/PC1vsPC2_16.png")
plt.close("all")
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 3',fontsize=20)
plt.ylabel('Principal Component - 4',fontsize=20)
plt.title(f"Total Explained Variance {sum(pca.explained_variance_ratio_[2:]):.3f}%")
targets = [i for i in range(10)]
for target in targets:
    indicesToKeep = (labels == target).values 
    if target<4:
        c = "r"
    elif 4<=target<7:
        c = "g"
    else:
        c = "b"
    plt.scatter(pc_values.loc[indicesToKeep, 'PC3']
               , pc_values.loc[indicesToKeep, 'PC4'], s = 40)

plt.legend(targets,prop={'size': 10})
plt.savefig("./03 Reports/PC3vsPC4_16.png")

plt.close("all")
