
# This article talks about training CIFAR 10 dataset with ResNet18 Architecture

This file is submitted as part of Assignment 8 for EVA6 Course


## Contributors

* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)
---
## Table of Contents
  - [Installation](#Installation)
  - [About the Model](#about-the-model)
  - [Techniques Used](#techniques-used)
  - [Graphs](#-graphs)
  - [Images](#-images)
  - [References](#references)
  

---
## Installation

We made a python library for using this our resources very easily. Just install it and call the function.

```
pip install woollylib
```

### Project Structure 

```bash
├── woollylib
│   ├── models
│   │     ├── model.py
│   │     └──  resnet.py
│   ├── utils
│   │    ├── gradcam
│   │    │   ├── _init_.py
│   │    │   ├── compute.py
│   │    │   ├── gradcam.py
│   │    │   └── util.py   
│   │    │
│   │    ├── transform.py 
│   │    ├── utils.py 
│   │    └── visualize.py 
│   │ 
│   ├── _init_.py
│   ├── backpropagation.py
│   ├── dataset.py
│   ├── main.py
│   ├── scheduler.py
│   └──training.py
│   
├── setup.py
├── LICENSE
├── CHANGELOG.txt
├── MANIFEST.IN
├── README.txt
├── requirements.txt
├── .gitignore
└── README.md
```



---
## About the Model 
* ### Introduction 
  After the first CNN-based architecture like (AlexNet) that win the ImageNet 2012 competition, Every subsequent winning architecture uses more layers in a deep neural network to reduce the error rate. This works for less number of layers, but when we increase the number of layers, there is a common problem in deep learning associated with that called Vanishing/Exploding gradient. This causes the gradient to become 0 or too large. Thus when we increases number of layers, the training and test error rate also increases.

  <image src='assets/Graph_Resnet.png'>

  In the above plot, we can observe that a 56-layer CNN gives more error rate on both training and testing dataset than a 20-layer CNN architecture, If this was the result of over fitting, then we should have lower training error in 56-layer CNN but then it also has higher training error. After analyzing more on error rate the authors were able to reach conclusion that it is caused by vanishing/exploding gradient.

  ResNet, which was proposed in 2015 by Microsoft introduced a new architecture called Residual Network.


* ### How does this model work ?
  
  <b> Residual Block </b>

  In order to solve the problem of the vanishing/exploding gradient, this architecture introduced the concept called Residual Network. In this network we use a technique called skip connections . The skip connection skips training from a few layers and connects directly to the output.

  The approach behind this network is instead of layers learn the underlying mapping, we allow network fit the residual mapping. So, instead of say H(x), initial mapping, let the network fit, ``` F(x) := H(x) – x which gives H(x) := F(x) + x``` .

  <image src='assets/Resnet_1.png'>

  The advantage of adding this type of skip connection is because if any layer hurt the performance of architecture then it will be skipped by regularization. So, this results in training very deep neural network without the problems caused by vanishing/exploding gradient.  The authors of the paper experimented on 100-1000 layers on CIFAR-10 dataset.

  There is a similar approach called “highway networks”, these networks also uses skip connection. Similar to LSTM these skip connections also uses parametric gates. These gates determine how much information passes through the skip connection. This architecture however  has not provide accuracy better than ResNet architecture.

  * <b> Network Architechture </b>

    This network uses a 34-layer plain network architecture inspired by VGG-19 in which then the shortcut connection is added. These shortcut connections then convert the architecture into residual network. 

    <image src='assets/Resnet_Arch.png'>

---
## Techniques Used 

* ### GradCam 
  ---
  * ### Introduction 
    CAM known as Grad-Cam. Grad-Cam published in 2017, aims to improve the shortcomings of CAM and claims to be compatible with any kind of architecture. This technique does not require any modifications to the existing model architecture and this allows it to apply to any CNN based architecture, including those for image captioning and visual question answering. For fully-convolutional architecture, the Grad-Cam reduces to CAM.
  * ### How do we solve this ?
    Grad-Cam, unlike CAM, uses the gradient information flowing into the last convolutional layer of the CNN to understand each neuron for a decision of interest. To obtain the class discriminative localization map of width u and height v for any class c, we first compute the gradient of the score for the class c, yc (before the softmax) with respect to feature maps Ak of a convolutional layer. These gradients flowing back are global average-pooled to obtain the neuron importance weights ak for the target class.

    <image src='assets/Weights_1.png'>

    After calculating ak for the target class c, we perform a weighted combination of activation maps and follow it by ReLU.

    <image src='assets/Linear_Combination.png' >

    This results in a coarse heatmap of the same size as that of the convolutional feature maps. We apply ReLU to the linear combination because we are only interested in the features that have a positive influence on the class of interest. Without ReLU, the class activation map highlights more than that is required and hence achieve low localization performance.

  * ### Pipeline Architechture 
    <image src='assets/Pipeline_arch.png' >

  * <b> Pytorch Implementation </b> 
      ```python
      Pending... 
      ```

* ### RandomCrop
  ---
    This Technique is used to crop a random part of the input.

    <b> Syntax </b> 
    ```python
    class albumentations.augmentations.crops.transforms.RandomCrop (height, width, always_apply=False, p=1.0) 
    ```
    Where Arguments are,
    * <b> Height </b> is a int value and it is used to declare the height of the crop.

    * <b> Width </b>is a int value and it is used to declare the width of the crop.

    * <b> Parameter(p) </b>  is probability of applying the transform which is by default "0".

    The final targets would be an Image or Mask or bboxes or keypoints with Type of the Image being uint8 or float32.
* ### CutOut  
  ---
    This Technique is used to CoarseDropout of the square regions in the image.

    <b> Syntax </b> 
    ```python
    class albumentations.augmentations.transforms.Cutout (num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5) 
    ```
    Where Arguments are,
    * <b> Num_Holes </b> is a int value and it is used to declare the number of regions to zero out.


    * <b> Max_h_Size</b> is a int value and it is used to set the maximum height of the hole.


    * <b> Max_w_Size </b> is a int value and it is used to set the maximum width of the hole.


    * <b> fill_value </b> can be a int,float of list of int,float values that can be used to defind the dropped pixels.

    The final targets would be an Image with Type of the Image being uint8 or float32.

* ### Rotate
  ---
    This Technique helps rotate the input by an angle selected randomly from the uniform distribution.
    <b> Syntax </b> 
    ```python
    class albumentations.augmentations.geometric.rotate.Rotate (limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)
    ```
    Where Arguments are,
    * <b> Limit</b> is a int value which can range from which a random angle is picked. If limit is a single int an angle is picked from (-limit, limit) if not then default it is (-90, 90)


    * <b>Interpolation</b> is a OpenCV flag value where the flag is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4 if not then default it is cv2.INTER_LINEAR.


    * <b> Border_Mode</b> is a OpenCV flag value where the flag is used to specify the pixel extrapolation method. Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101 if not then default it is  cv2.BORDER_REFLECT_101


    * <b> Value </b> can be int or float or list of integers or list of float values which is a padding value if border_mode is cv2.BORDER_CONSTANT.


    * <b> Mask_Value</b> can be int or float or list of integers or list of float values which is a padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.


    * <b> Parameter(p) </b>  is probability of applying the transform which is by default "0".

    The final targets would be an image or Mask or Bboxes ir keypoints with type of the image being uint8 or float32.
    
* ### ReduceLROnPlateau (Optimizer Technique)
  ---
  This Technique is used when a metric has stopped improving then we reduce learning rate.Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.

  <b> Syntax </b> 
  ```python
  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
  ```
  Where Arguments are,
  
    * <b> Optimizer </b> takes the args Optimizer and it is Wrapped optimizer.


    * <b>Mode</b> is a float value which can be one of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing and then in max mode it will be reduced when the quantity monitored has stopped increasing if not then default is ‘min’.


    * <b>Factor</b> is a float value which can factor the learning rate by reducing. new_lr = lr * factor and by default it takes the value 0.1.


    * <b> Patience</b> is a int value which tell us the number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then and by default it takes the value 10.


    * <b>Threshold </b> is a float value which tell us the threshold for measuring the new optimum, to only focus on significant changes and by default it takes the value 1e-4.


    * <b>Threshold Mode</b> is a String value which tell us One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode and by default it takes the value "rel".


    * <b>Cooldown</b>  is a int value which tells us the number of epochs to wait before resuming normal operation after lr has been reduced and by default it takes the value 0.


    * <b>Minimum Learing Rate (min_lr) </b> is a float value or a list which tells us a scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively and by default it takes the value 0.


    * <b> eps (float)</b> is a float value which tells us the minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored and by default it takes the value 1e-8.


    * <b>Verbose </b> is a Boolean value which tells us If True, prints a message to stdout for each updatede fault it takes the value False.



---
## Graphs 
---

## Images
---
## References

* https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

