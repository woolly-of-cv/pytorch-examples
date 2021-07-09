# This article talks about training of custom ResNet architecture for CIFAR 10

This file is submitted as part of Assignment 9 for EVA6 Course

**Highlights**

## Contributors

* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)

---

### Table of Contents
- [Installation](#installation)
- [About the Model](#about-the-model)
- [Techniques Used](#techniques-used)
  - [GradCam](#gradcam)
  - [Custom Resnet](#custom-resnet)
  - [Loss Smooth](#loss-smooth)
  - [Grad Scaler](#grad-scaler)
- [Graphs](#graphs)
- [Images](#images)
- [References:](#references)

___
### Installation

We made a python library for using this our resources very easily. Just install it and call the function.

```
pip install woollylib
```

URL : https://pypi.org/project/woollylib/

### Project Structure 

```
├── woollylib
│   ├── bp
│   │   ├── autocast
│   │   │     └── backpropagation.py   
│   │   ├── losses
│   │   │     └── backpropagation.py  
│   │   ├── optimizer
│   │   │     └── backpropagation.py  
│   │   ├── ricap
│   │   │     └── backpropagation.py   
│   │   └── vanila  
│   │         └── backpropagation.py   
│   │ 
│   ├── models
│   │   ├── custom
│   │   │     └── custom_resnet.py 
│   │   ├── model.py
│   │   └── resnet.py
│   │ 
│   ├── preloading
│   │    └── dataset.py 
│   │ 
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
│   ├── dataset.py
│   ├── main.py
│   ├── scheduler.py
│   └──training.py
│   
├── setup.py
├── LICENSE
├── MANIFEST.IN
├── README.txt
├── requirements.txt
├── .gitignore
└── README.md
```


___

### About the Model

**Overview:**

---
### Techniques Used


#### GradCam

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


#### Custom Resnet


#### Loss Smooth

<image src='assets/Loss.png' >


#### Grad Scaler

<image src='assets/gradscaler.png' >

---

### Graphs

---

### Images

---

### References:

---
