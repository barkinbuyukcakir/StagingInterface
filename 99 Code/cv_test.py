import os
os.chdir("/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/VisualAttentionProjects/DentalStagingVIT")
from datatypes import AttentionDatabase,Metrics
import pickle
import pandas as pd
from torch import zeros,vstack,Tensor,zeros_like
from torch.nn import Softmax
from tqdm import tqdm
import matplotlib.pyplot as plt


class CvTester():
    """
    Generated the mean and weighted mean attention maps, records metrics
    """
    def __init__(self,visualize = True, silent = False) -> None:
        pass
        self.visualize = visualize
        self.silent = silent
    
        pass
    
    def parse(self,name):
        """
        Given directory name, returns:
        ModelId, ModelType, patch_size, layers, heads, tooth, CLAHE, RandomAffine
        """
        ps, tooth, improvement, *modelId = name.split('_')
        clahe = True if improvement.__contains__("CL") else False
        randomAffine = True if improvement.__contains__("RA") else False
        #For backwards compatibility on naming convention
        try:
            patch_size,n_layers,n_heads = ps.split("-")
        except ValueError:
            patch_size = ps
            n_layers = 12
        return (f"{modelId[0]}_{modelId[1]}",
                patch_size[:-2],
                int(patch_size[-2:]),
                int(n_layers),
                int(n_heads),
                int(tooth),
                clahe,
                randomAffine
            )

    def test(self):
        try:
            df = pd.read_excel("./03 Reports/Summary.xlsx",header=0,index_col=[i for i in range(11)])
            createDf = False
        except FileNotFoundError:
            createDf = True

 
        models = os.listdir('./02 Results/test_results')
        models = [i for i in models if i.__contains__("1106")]
        pbar = tqdm(models,disable=self.silent)
        for model in pbar:
            model_id =f'{model.split("_")[-2]}_{model.split("_")[-1]}'
            if not createDf:
                if model_id in df.index.get_level_values(0):
                    continue         
            path = f"./03 Reports/mean_attention_maps/{model}"
            os.makedirs(path,exist_ok=True)
            folds = [i for i in os.listdir(f'./02 Results/test_results/{model}')]
            n_folds = len(folds)
            metadata = self.parse(model)
            ims = dict([(0,[]),(1,[]),(2,[]),(3,[]),(4,[]),(5,[]),(6,[]),(7,[]),(8,[]),(9,[])])
            masks = dict([(0,[]),(1,[]),(2,[]),(3,[]),(4,[]),(5,[]),(6,[]),(7,[]),(8,[]),(9,[])])
            results = dict([(0,[]),(1,[]),(2,[]),(3,[]),(4,[]),(5,[]),(6,[]),(7,[]),(8,[]),(9,[])]) #Images, Masks, Foci
            preds = dict([(0,[]),(1,[]),(2,[]),(3,[]),(4,[]),(5,[]),(6,[]),(7,[]),(8,[]),(9,[])])
    
            #Accumulate all images,masks,foci and predictions across folds
            pbar.set_description(f"Testing model {model}")
            df2 = pd.DataFrame(columns=['ModelId','ModelType','PatchSize','Layers','Heads','Tooth','CLAHE','RandomAffine','MeanAccStd','AverageAttentionsPath','Fold','Accuracy','CohenKappa','MAE','MSE',])
            
            for foldNum, fold in enumerate(folds):
                with open(f"./02 Results/test_results/{model}/{fold}",'rb') as f:
                    dbase = pickle.load(f)
                im_size = [*dbase.images[0].shape[1:]]
                try:
                    _ = dbase.metrics
                except AttributeError:
                    dbase.metrics = Metrics(dbase.predictions,dbase.labels)
                for i in range(len(dbase)):
                    im,mask,result,pred, label = dbase.rollout(i,visualize=False)
                    ims[label]+=[im]
                    masks[label]+=[mask.permute(2,0,1)]
                    results[label]+=[result.permute(2,0,1)]
                    preds[label]+=[pred]
                ind = len(df2)
                df2.loc[ind] = [*metadata,None,path,foldNum+1,*dbase.metrics()]
                del dbase
                
                
            #For all labels, find the average and the weighted average of the attention maps
            mean_acc = df2[df2.ModelId == model_id].Accuracy.mean()
            mean_std = df2[df2.ModelId == model_id].Accuracy.std()
            df2.loc[(df2.ModelId == model_id),"MeanAccStd"] = f"{mean_acc:.3f} ({mean_std:.3f})"

            multiindex = pd.MultiIndex.from_frame(df2.loc[:,df2.columns[:11]],names = df2.columns[:11])
            df2.drop(columns=df2.columns[:11],inplace=True)
            df2.index = multiindex
            if createDf:
                createDf=False
                df = df2
                # df2.to_excel("./03 Reports/Summary.xlsx")
            else:
                df = pd.concat([df,df2],ignore_index=False)
                # df.to_excel("./03 Reports/Summary.xlsx")
                pbar.set_description("Report out.")
            pbar.set_description("Plotting")
            for i in range(10):
                try:
                    atts = vstack(masks[i])
                except RuntimeError:
                    continue
                #Mean
                mean = atts.mean(dim=0)
                #Accuracy-Weighted mean
                p = Tensor(preds[i])
                weigths = Softmax(dim=0)((p-(zeros_like(p)+2)).abs())
                w_avg = zeros_like(atts[0])
                for l in range(len(preds[i])):
                    w_avg += atts[l]*weigths[l]
                fig,axs = plt.subplots(1,3,figsize = (36,12))
                fig.suptitle(f"Stage {i} Mean Attention Maps")
                axs[0].imshow(vstack(ims[i]).mean(dim=0),cmap="gray")
                axs[0].axis("off")
                axs[1].imshow(mean.numpy(),cmap="jet")
                axs[1].set_title("Mean")
                axs[1].axis("off")
                axs[2].imshow(w_avg.numpy(),cmap = "jet")
                axs[2].set_title("Accuracy Weighted")
                axs[2].axis("off")
                fig.savefig(f"./03 Reports/mean_attention_maps/{model}/stage_{i}_comp.png")
                extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(f"./03 Reports/mean_attention_maps/{model}/stage_{i}_imMean.png",bbox_inches = extent)
                extent = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(f"./03 Reports/mean_attention_maps/{model}/stage_{i}_atMean.png",bbox_inches = extent)
                extent = axs[2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(f"./03 Reports/mean_attention_maps/{model}/stage_{i}_wMean.png", bbox_inches = extent)
                if self.visualize:
                    fig.show()
                plt.close()
                
        df.to_excel("./03 Reports/Summary.xlsx")
                # 
                # 'ModelId','ModelType','PatchSize','Layers','Heads','Tooth','CLAHE','RandomAffine','Fold','AverageAttentionsPath','Accuracy','CohenKappa','MAE','MSE'
                


if __name__ == '__main__':
    CvTester(visualize=False).test() 