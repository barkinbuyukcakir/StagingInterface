# from collections.abc import Iterable
from PIL import Image
import torch
import torchvision.transforms as T
from dataclasses import dataclass
from typing import Any, Callable, Tuple, Sequence
import numpy as np
from sklearn.metrics import accuracy_score,cohen_kappa_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from cv2 import resize
from sys import float_info
eps = float_info.epsilon

@dataclass
class AttentionReturn:
    """
    Class for structured return of attention matrices with additional information.

    Args:
        image (`torch.Tensor`):
            The original image at the requested index.
        prediction (`int`):
            The prediction for the image at index.
        label (`int`):
            The true label for the image at given index.
        attention (`torch.Tensor`): 
            The attention matrix, either reduced across heads or complete.
        
    """
    image: torch.Tensor
    prediction: int
    label: int
    attention: torch.Tensor

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return(self.image,self.prediction,self.label,self.attention)

@dataclass
class AttentionVisualReturn():
    """
    Class for structured return of attention matrix visualizations.

    Args:
        image (`PIL.Image`):
            The original image at the requested index.
        prediction (`int`):
            The prediction for the image at index.
        label (`int`):
            The true label for the image at given index.
        attention_image (`Tuple[PIl.Image]`): 
            The attention images collated.
        
    """
    image: Image
    prediction: int
    label: int
    attention_image: Sequence[Image.Image]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return(self.image,self.prediction,self.label,self.attention_image)

class AttentionDatabase():
    def __init__(
            self,
            attentions:torch.Tensor,
            predictions: torch.Tensor,
            labels: torch.Tensor,
            images: torch.Tensor
    ) -> None:
        """
        Stores the attentions in the shape :math:`(B,L,N,H,W)`,
        where `B` = Batch size,
              `L` = Layer number,
              `N` = Head number,
              `H`,`W` = Height, Width 
        """
        self.num_attention_heads = attentions[0].size(2)
        self.attentions = torch.vstack(attentions).cpu()
        self.predictions = predictions.cpu()
        self.labels = labels.cpu()
        self.images = images
        
        self.ToPILImage = T.ToPILImage()

        self.metrics = Metrics(preds=predictions, labels= labels)

    def __len__(self):
            return self.attentions.size(0)
    def __getitem__(self,index):
        return self.attentions[index]        
    
    def get(
            self,
            index:int,
            pil_image:bool=False
            ) -> AttentionReturn:
        """
        Return the prediction, label and attention matrix for all layers and heads for the given index.

        """
        image = self.images[index]
        prediction= self.predictions[index]
        label = self.labels[index]
        attentions = self.attentions[index]
        return AttentionReturn(image,prediction.item(),label.item(),attentions)
    
    def get_reduced(
            self,
            index:int,
            reduce_fn:Callable[[torch.Tensor],torch.Tensor]=torch.mean,
            pil_image:bool=False
            ) -> AttentionReturn:
        """
        Return the reduced attention matrix across heads.
        Reduction is defined by the `reduce_fn`.
        """
        image = self.images[index]
        prediction = self.predictions[index]
        label = self.labels[index]
        attentions = reduce_fn(self.attentions[index],dim=1)
        return AttentionReturn(
            image,
            prediction.item(),
            label.item(),
            attentions
        )
    def visualize_attention_map(
            self,
            index:int,
            reduce_fn:Callable[[torch.Tensor],torch.Tensor]=torch.mean,
            drop_threshold:float = 0
            ) -> Image:
        """
        Visualize the attention matrices for the image at the given index.

        The attention matrices are reduced across heads with `reduce_fn`.
        All layers are presented in a grid image.    
        """
        out = self.get_reduced(index=index,reduce_fn=reduce_fn)
        image = self.ToPILImage(out.image)
        prediction = out.prediction
        label =out.label
        attentions = out.attention

        # Get the first rows of all layers and drop the class token.
        # Reshape all rows to 14x14
        n_heads,n_layers = attentions.shape
        attentions = attentions[:,0,1:]
        #Collate
        attention_images = []
        for i in attentions:
            attention_images.append(
                Image.fromarray(i.numpy(),"L").resize((120,120))
            )


        return AttentionVisualReturn(
            image=image,
            prediction=prediction,
            label=label,
            attention_image=attention_images
        )

    @torch.no_grad()
    def rollout(self,index:int,reduction: str | Callable = "max", discard_ratio:float=0.9,**kwargs:Any) -> Tuple:
        """
        Outputs the attention rollout for a test case.
        Args:
            - index (`int`): 
                Which sample to perform the rollout for.
            - reduction (`str` | `Callable`):
                How to reduce attention matrix across heads. Can be `"mean"`,`"min"`, `"max"` or custom `Callable` which operates on `dim=1`.
            - discard_ratio (`float`):
                To be implemented. For the discard of attention values below a certain threshold.
        
        """
        image,pred,label, attentions = self.get(index)()
        if len(self.attentions.shape) > 5:
            attentions = attentions[0]
        add_label_text = kwargs.pop("add_label_text",True)
        add_pred_text = kwargs.pop("add_pred_text",True)
        visualize = kwargs.pop("visualize",True)
        grid_size = kwargs.pop("grid_size",None)

        #Reduce attention matrix across heads
        
        if isinstance(reduction,str):
            if reduction == "mean":
                att_mat = torch.mean(attentions,dim=1)
            elif reduction == "min":
                att_mat = torch.min(attentions,dim=1)[0]
            elif reduction == "max":
                att_mat = torch.max(attentions,dim=1)[0]
        elif isinstance(reduction,Callable):
            att_mat = reduction(attentions)
        
        #Flat representation of the attention matrix
        flat = att_mat.view(att_mat.size(0),-1)
        #Dropping the lower attention values
        _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
        indices = indices[indices != 0]
        flat[0, indices] = 0

        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)


        joint_attentions = torch.zeros(aug_att_mat.size()) 
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1,aug_att_mat.size(0)):
            flat = aug_att_mat[n].view(1,-1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            joint_attentions[n] = torch.matmul(aug_att_mat[n],joint_attentions[n-1])
        
        v= joint_attentions[-1]
        if grid_size is None:
            grid_size = (int(np.sqrt(aug_att_mat.size(-1))),int(np.sqrt(aug_att_mat.size(-1))))
        mask = v[0,1:].reshape(grid_size).detach().numpy()
        mask = resize(mask/(mask.max()+eps), (image.size(-1),image.size(-2)))[...,np.newaxis]
        result = (mask * image.permute(1,2,0).numpy())
        if visualize:
            fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (16,16))

            ax1.set_title("Original")
            ax2.set_title("Attention Map")
            ax3.set_title("Model Focus")
            ax1.imshow(image.permute(1,2,0).numpy()/255 if image.max()>1 else image.permute(1,2,0).numpy(),cmap="gray")
            if add_label_text:
                ax1.text(140,200,f"Ground truth: {label}",c="white")
            if add_pred_text:
                ax1.text(140,210,f"Prediction: {pred}",c="white")
            ax2.imshow(image.permute(1,2,0).numpy()/255 if image.max()>1 else image.permute(1,2,0).numpy(),cmap="gray")
            ax2.imshow(mask,cmap = "jet",alpha =0.4)
            ax3.imshow(result,cmap ='gray')
            print(f"Predicted: {pred} | Label: {label}")
            fig.show()
        return image,torch.Tensor(mask),torch.Tensor(result), pred,label


class Metrics():
    def __init__(self, preds, labels) -> None:
        self.preds = preds
        self.labels = labels
        
    def __call__(self):
        return [getattr(self,i) for i in dir(self) if i.startswith("_metric")]
    @property
    def _metric_accuracy(self):
        return accuracy_score(self.labels,self.preds)
    
    @property
    def _metric_cohen_kappa(self):
        return cohen_kappa_score(self.labels,self.preds)

    @property
    def _metric_mae(self):
        return mean_absolute_error(self.labels,self.preds)

    @property
    def _metric_mse(self):
        return mean_squared_error(self.labels,self.preds)

