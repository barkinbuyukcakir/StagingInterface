import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch import vstack, uint8, Tensor
from psd_tools import PSDImage
from torchvision.transforms import ToTensor,ToPILImage
from cv2 import createCLAHE,resize
from skimage.exposure import rescale_intensity


class CustomDatasetStaging(Dataset):
    def __init__(self,annotations: str | pd.DataFrame, transform = None,patch_size: int = 32,**kwargs) -> None:
        super().__init__()
        assert (type(annotations) == str) or (type(annotations) == pd.DataFrame), 'Annotations should be provided as excel file or DataFrame'
        if type(annotations) == str:
            assert annotations.endswith(".xlsx"),"Annotation file must be .xlsx"
            self.annot = pd.read_excel(annotations)
        elif type(annotations) == pd.DataFrame:
            self.annot = annotations
        self.clahe = createCLAHE(clipLimit = kwargs.pop("clip_limit",3), 
                                 tileGridSize = kwargs.pop("tile_size",(4,4)))
        self.do_clahe = kwargs.pop("clahe",False)
        self.per_image_norm = kwargs.pop("per_image_norm",False)
        self.transform = transform
        self.patch_size = patch_size
        self.n_channels = kwargs.pop("n_channels",3)
        self.range = kwargs.pop("range","1")

    def __len__(self):
        return len(self.annot)
    
    def doClahe(self,img):
        s1,s2 = img.shape
        s1 += self.patch_size- s1%self.patch_size
        s2 += self.patch_size - s2%self.patch_size
        img = resize(img,(s2,s1))
        img = self.clahe.apply(img)
        return ToTensor()(img)

    def __perImageNorm(self,img):
        MAX,MIN = img.max(),img.min()
        return (img-MIN)/(MAX-MIN)


    def __getitem__(self, index):
        img_path = self.annot.iloc[index,0]
        if img_path.endswith('.psd'):
            image = PSDImage.open(img_path).composite()

            image = image.convert("L")
            if self.do_clahe:
                image = self.doClahe(np.array(image))
            else:
                image = ToTensor()(image)
        else:
            image = read_image(img_path)[:3]
            if self.do_clahe:
                image = self.doClahe(image.permute(1,2,0)[:,:,0].to(uint8).numpy())
            image = image/255
        img_label = self.annot.iloc[index,-2]
        if image.shape[0] == 1:
            image = vstack((image,image,image))
        if self.transform:
            image = self.transform(image)
        if self.per_image_norm:
            image = self.__perImageNorm(image)
        if self.range =="255":
            image = image*255
        if self.n_channels ==3:
            return vstack([image,image,image]), int(img_label)
        else:
            return image, int(img_label)


class CustomDatasetTriplets(Dataset):
    def __init__(self,annotations: str | pd.DataFrame, transform = None,patch_size: int = 32,**kwargs):
        super().__init__()
        assert (type(annotations) == str) or (type(annotations) == pd.DataFrame), 'Annotations should be provided as excel file or DataFrame'
        if type(annotations) == str:
            assert annotations.endswith(".xlsx"),"Annotation file must be .xlsx"
            self.annot = pd.read_excel(annotations)
        elif type(annotations) == pd.DataFrame:
            self.annot = annotations
        self.per_image_norm = kwargs.pop("per_image_norm",False)
        self.transform = transform
        self.n_channels = kwargs.pop("n_channels",1)
        self.intensity = kwargs.pop('intensity',False)

    def __perImageNorm(self,img):
        MAX,MIN = img.max(),img.min()
        return (img-MIN)/(MAX-MIN)

    def load_and_process_image(self,img_path):
        
        if img_path.endswith('.psd'):
            image = PSDImage.open(img_path).composite()

            image = image.convert("L")
            image = ToTensor()(image)
        else:
            image = read_image(img_path)[:3]
            image = image/255 if image.max()>1 else image
        
        if self.transform:
            image = self.transform(image)
        if self.intensity:
            p3,p97 = np.percentile(image[0],(3,97))
            image = rescale_intensity(image[0].numpy(),in_range=(p3,p97))
            image = Tensor(image).unsqueeze(0)
        if self.per_image_norm:
            image = self.__perImageNorm(image)
        if self.n_channels ==3:
            return vstack([image,image,image])
        else:
            if image.size(0)==1:
                return image
            elif image.size(0) == 3:
                return image[0,:,:].unsqueeze(0)
        
    def get_random_positive(self,label):
        matches = self.annot[self.annot.iloc[:,-2] == label.item()]
        try:
            path = matches.sample(n=1).iloc[0,0]
            return self.load_and_process_image(path)
        except:
            path = self.annot.sample(n=1).iloc[0,0]
            return self.load_and_process_image(path)
    def get_random_negative(self,label):
        matches = self.annot[self.annot.iloc[:,-2] != label.item()]
        try:
            path = matches.sample(n=1).iloc[0,0]
            return self.load_and_process_image(path)
        except:
            path = self.annot.sample(n=1).iloc[0,0]
            return self.load_and_process_image(path)
    
    def __getitem__(self, index):
        """
        Returns (Anchor, Positive, Negative, Anchor Label)
        """
        img_path = self.annot.iloc[index,0]
        # positive_path = self.__findPositive(index)
        # negative_path,neg_label = self.__findNegative(index)
        img_label = self.annot.iloc[index,-2]
        return self.load_and_process_image(img_path),img_label
    def __len__(self):
        return len(self.annot)



if __name__ == "__main__":
    #TODO: add isolated testability here (I'm probably never doing it)
    pass