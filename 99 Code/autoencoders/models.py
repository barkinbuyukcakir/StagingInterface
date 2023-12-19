import torch.nn as nn
import torch
import torch.nn.functional as F


class LatentClassifier(nn.Module):
    def __init__(self,n_classes,embed_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim,100),
            nn.ReLU(),
            nn.Linear(100,n_classes)
        )
    def forward(self,x):
        return(self.classifier(x))
    
class ConvAutoencoder(nn.Module):
    def __init__(self, n_channels,embed_dim,out_range = 1):
        super().__init__()
        self.out_range = out_range
        self.encoder_block = nn.Sequential(
            nn.Conv2d(n_channels,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )
        self.to_latent = nn.Sequential(
            nn.Linear(100352,1000),
            nn.ReLU(),
            nn.Linear(1000,embed_dim),
            
        )
        
        self.from_latent = nn.Sequential(
            nn.Linear(embed_dim,1000),
            nn.ReLU(),
            nn.Linear(1000,100352),
        )
        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(128,128,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),           
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid(),
        )

    def forward(self,x,return_embeddings = False):
        x = self.encoder_block(x)
        embeddings = self.to_latent(x.flatten(1))
        bs = x.size(0)
        x = self.from_latent(F.normalize(embeddings,p=2))
        x = self.decoder_block(x.view(bs,128,28,28))
        x= self.out_conv(x)
        if return_embeddings:
            return x, embeddings
        else:
            return x
        

class ConvAutoencoderSVD(nn.Module):
    def __init__(self, n_channels,embed_dim,svd_dim=10,out_range = 1):
        super().__init__()
        self.out_range = out_range
        self.svd_dim = svd_dim
        self.encoder_block = nn.Sequential(
            nn.Conv2d(n_channels,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )
        self.to_latent = nn.Sequential(
            nn.Linear(100352,5000),
            nn.ReLU(),
            nn.Linear(5000,1000),
            nn.ReLU(),
            nn.Linear(1000,embed_dim),
            
        )
        
        self.from_latent = nn.Sequential(
            nn.Linear(svd_dim,1000),
            nn.ReLU(),
            nn.Linear(1000,5000),
            nn.ReLU(),
            nn.Linear(5000,100352//2),
        )
        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),           
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid(),
        )

    def forward(self,x,return_embeddings=False):
        x = self.encoder_block(x)
        embeddings = self.to_latent(x.flatten(1))
        bs = x.size(0)
        U,S,V = torch.linalg.svd(embeddings,full_matrices = False)
        embeddings = U[:,:self.svd_dim]@torch.diag_embed(S[:self.svd_dim])@V[:self.svd_dim,:self.svd_dim]
        del U,S,V
        x = self.from_latent(F.normalize(embeddings))
        x = self.decoder_block(x.view(bs,64,28,28))
        x= self.out_conv(x)
        if return_embeddings:
            return x, embeddings
        else:
            return x