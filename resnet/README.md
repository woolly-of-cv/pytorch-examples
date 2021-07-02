
# This article talks about training CIFAR 10 dataset with ResNet18 Architecture

This file is submitted as part of Assignment 8 for EVA6 Course

### Highlights


---
## Contributors

* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)
---
## Table of Contents
  - [About the Model](#about-the-model)
  - [Techniques Used](#techniques-used)
  - [Graphs](#-graphs)
  - [Images](#-images)
  - [References](#references)
  

---
## About the Model 
* ### Introduction 
* ### How does this model work ?
  

---
## Techniques Used 

* ### GradCam 
  ---

    <b> Syntax </b> 
    ```python
    class GradCAM(nn.Module):
        def __init__(self, model):
            super(GradCAM, self).__init__()
            
            # get the pretrained network
            self.wy = model
            
            # disect the network to access its last convolutional layer
            self.features_conv = self.wy.feature
            
            # get the classifier of the model
            self.classifier = self.wy.classifier
            
            # placeholder for the gradients
            self.gradients = None
            
            self.classes = self.wy.classes
        
        # hook for the gradients of the activations
        def activations_hook(self, grad):
            self.gradients = grad
            
        def forward(self, x):
            x = self.features_conv(x)
            
            # register the hook
            h = x.register_hook(self.activations_hook)
            
            # apply the remaining pooling
            x = self.classifier(x)

            x = x.view(-1, self.classes)
            
            return x
        
        # method for the gradient extraction
        def get_activations_gradient(self):
            return self.gradients
        
        # method for the activation exctraction
        def get_activations(self, x):
            return self.features_conv(x)
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

