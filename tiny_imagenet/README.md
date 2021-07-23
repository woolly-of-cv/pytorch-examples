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
  
  
## BottleNeck:
  
## Graphs:
  
  <image src='assets/TinyImagenet.png'>
    
## Gradcam Images:
    
  <image src='assets/GradCam.png'>
  
## Misclassified Predictions:
    
   <image src='assets/MisclassifiedPredictions.png'>
     

## References:
  
