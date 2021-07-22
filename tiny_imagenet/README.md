Tiny Imagenet

## Contributors

* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)
---

## Table of Contents
  - [Tiny Imagenet Dataset](#tiny-imagenet-dataset)
  - [About the Model](#about-the-model)
  - [Graphs](#graphs)
  - [Images](#images)
  - [References](#references)
  
## Tiny Imagenet Dataset:

<p> It is a smaller version derived from the monolith ImageNet challenge. The dataset consists of 100000 images of 200 classes (500 for each class) downsized to 64×64 colored images. Each class has 500 training images, and in total 1000 test images. The download is done using the script in the https://github.com/woolly-of-cv/pytorch-lib/tree/main/woollylib/dataset/tiny_imagenet </p>

### Preprocessing:

<p> The labels text file consist of the class names. The folders consisting of the training images are then renamed to their respective class names. Custom dataset class is created to read the image and label and for image transforms. <p>

## About the Model:
  
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 64, 64]           1,728
          WyConv2d-2           [-1, 64, 64, 64]               0
       BatchNorm2d-3           [-1, 64, 64, 64]             128
            Conv2d-4           [-1, 64, 64, 64]          36,864
          WyConv2d-5           [-1, 64, 64, 64]               0
       BatchNorm2d-6           [-1, 64, 64, 64]             128
        WyResidual-7           [-1, 64, 64, 64]               0
            Conv2d-8          [-1, 128, 64, 64]          73,728
          WyConv2d-9          [-1, 128, 64, 64]               0
        MaxPool2d-10          [-1, 128, 32, 32]               0
      BatchNorm2d-11          [-1, 128, 32, 32]             256
             ReLU-12          [-1, 128, 32, 32]               0
           Conv2d-13          [-1, 128, 32, 32]         147,456
         WyConv2d-14          [-1, 128, 32, 32]               0
      BatchNorm2d-15          [-1, 128, 32, 32]             256
           Conv2d-16          [-1, 128, 32, 32]         147,456
         WyConv2d-17          [-1, 128, 32, 32]               0
      BatchNorm2d-18          [-1, 128, 32, 32]             256
       WyResidual-19          [-1, 128, 32, 32]               0
           Conv2d-20          [-1, 128, 32, 32]         147,456
         WyConv2d-21          [-1, 128, 32, 32]               0
      BatchNorm2d-22          [-1, 128, 32, 32]             256
           Conv2d-23          [-1, 128, 32, 32]         147,456
         WyConv2d-24          [-1, 128, 32, 32]               0
      BatchNorm2d-25          [-1, 128, 32, 32]             256
       WyResidual-26          [-1, 128, 32, 32]               0
          WyBlock-27          [-1, 128, 32, 32]               0
           Conv2d-28          [-1, 256, 32, 32]         294,912
         WyConv2d-29          [-1, 256, 32, 32]               0
        MaxPool2d-30          [-1, 256, 16, 16]               0
      BatchNorm2d-31          [-1, 256, 16, 16]             512
             ReLU-32          [-1, 256, 16, 16]               0
           Conv2d-33          [-1, 256, 16, 16]         589,824
         WyConv2d-34          [-1, 256, 16, 16]               0
      BatchNorm2d-35          [-1, 256, 16, 16]             512
           Conv2d-36          [-1, 256, 16, 16]         589,824
         WyConv2d-37          [-1, 256, 16, 16]               0
      BatchNorm2d-38          [-1, 256, 16, 16]             512
       WyResidual-39          [-1, 256, 16, 16]               0
           Conv2d-40          [-1, 256, 16, 16]         589,824
         WyConv2d-41          [-1, 256, 16, 16]               0
      BatchNorm2d-42          [-1, 256, 16, 16]             512
           Conv2d-43          [-1, 256, 16, 16]         589,824
         WyConv2d-44          [-1, 256, 16, 16]               0
      BatchNorm2d-45          [-1, 256, 16, 16]             512
       WyResidual-46          [-1, 256, 16, 16]               0
          WyBlock-47          [-1, 256, 16, 16]               0
           Conv2d-48          [-1, 512, 16, 16]       1,179,648
         WyConv2d-49          [-1, 512, 16, 16]               0
        MaxPool2d-50            [-1, 512, 8, 8]               0
      BatchNorm2d-51            [-1, 512, 8, 8]           1,024
             ReLU-52            [-1, 512, 8, 8]               0
           Conv2d-53            [-1, 512, 8, 8]       2,359,296
         WyConv2d-54            [-1, 512, 8, 8]               0
      BatchNorm2d-55            [-1, 512, 8, 8]           1,024
           Conv2d-56            [-1, 512, 8, 8]       2,359,296
         WyConv2d-57            [-1, 512, 8, 8]               0
      BatchNorm2d-58            [-1, 512, 8, 8]           1,024
       WyResidual-59            [-1, 512, 8, 8]               0
           Conv2d-60            [-1, 512, 8, 8]       2,359,296
         WyConv2d-61            [-1, 512, 8, 8]               0
      BatchNorm2d-62            [-1, 512, 8, 8]           1,024
           Conv2d-63            [-1, 512, 8, 8]       2,359,296
         WyConv2d-64            [-1, 512, 8, 8]               0
      BatchNorm2d-65            [-1, 512, 8, 8]           1,024
       WyResidual-66            [-1, 512, 8, 8]               0
          WyBlock-67            [-1, 512, 8, 8]               0
AdaptiveAvgPool2d-68            [-1, 512, 1, 1]               0
             View-69                  [-1, 512]               0
           Linear-70                  [-1, 200]         102,600
================================================================
Total params: 14,085,000
Trainable params: 14,085,000
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 59.51
Params size (MB): 53.73
Estimated Total Size (MB): 113.29
----------------------------------------------------------------
  
  
## BottleNeck:
  
## Graphs:
  
## Images:

## References:
  
