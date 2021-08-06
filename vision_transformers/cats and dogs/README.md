
# Classifying Dogs and Cats with Vision Transformers

### This file is submitted as part of Assignment 13 for EVA 6 Course.


## Table of Contents

  - [Diving Deep into Vision Transformers](#diving-deep-into-vision-transformers)
  - [Results](#results)
  - [Referances](#referances)


## Diving Deep into Vision Transformers

* ### <b>Revisiting the Theory </b>

    
  <p align="center">
    <img src='assets/ViT.png'>
    <b>Overview Architechture</b><br>
  </p>
  
  Let's take the same architechture that we periously did and with rohan's photo. An overview of the model is shown in the figure above. The standard Transformer receives as input a 1D
  sequence of token embeddings. To handle 2D images, we reshape the image x ∈ RH×W×C into a sequence of flattened 2D patches ```xp ∈ RN×(P2·C)```, where ```(H,W)``` is the resolution of the original
  image,Cisthenumberofchannels,```(P,P)```istheresolutionofeachimagepatch,andN =HW/P2 is the resulting number of patches, which also serves as the effective input sequence length for the Transformer. The Transformer uses constant latent vector size D through all of its layers, so we flatten the patches and map to D dimensions with a trainable linear projection (Eq. 1). We refer to the output of this projection as the patch embeddings.
  
  Similar to BERT’s [class] token, we prepend a learnable embedding to the sequence of embed- ded patches ```(z0 = xclass)```, whose state at the output of the Transformer encoder ```(z0L)``` serves as the image representation y. Both during pre-training and fine-tuning, a classification head is at- tached to z0L. The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.

  Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings. The resulting sequence of embedding vectors serves as input to the encoder.

  The MLP contains two layers with a GELU non-linearity.
  <p align="center">
    <img src='assets/MLP_Equations.png'>

  </p>


  * ### Inductive Bias

    Inductive bias. We note that Vision Transformer has much less image-specific inductive bias than CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model. In ViT, only MLP layers are local and transla- tionally equivariant, while the self-attention layers are global. The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution (as de- scribed below). Other than that, the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.

  * ### Hybrid Architecture

    As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN . In this hybrid model, the patch embedding projection ```E``` (from Eq 1) is applied to patches extracted from a CNN feature map. As a special case, the patches can have spatial size ```1x1```, which means that the input sequence is obtained by simply flattening the spatial dimensions of the feature map and projecting to the Transformer dimension. The classification input embedding and position embeddings are added as described above.

 * ### <b>Wait !! We should divive deep into what each class does in vision transformer</b>



## Results 

* ### Graphs 
* ### Images

## Referances
1. https://arxiv.org/abs/2010.11929v2

