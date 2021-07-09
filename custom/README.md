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
  - [Label Smoothing](#label-smoothing)
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


#### Label Smoothing

  * ### What is label smoothing
	  When using deep learning models for classification tasks, we usually encounter the following problems: overfitting, and overconfidence. Overfitting is well studied and can be tackled with early stopping, dropout, weight regularization etc. On the other hand, we have less tools to tackle overconfidence. Label smoothing is a regularization technique that addresses both problems.

  * ### Overconfidence and Calibration
	A classification model is calibrated if its predicted probabilities of outcomes reflect their accuracy. For example, consider 100 examples within our dataset, each with predicted probability 0.9 by our model. If our model is calibrated, then 90 examples should be classified correctly. Similarly, among another 100 examples with predicted probabilities 0.6, we would expect only 60 examples being correctly classified.
	 #### Model calibration is important for
	* model interpretability and reliability
	* deciding decision thresholds for downstream applications
	* integrating our model into an ensemble or a machine learning pipeline
	
	An overconfident model is not calibrated and its predicted probabilities are consistently higher than the accuracy. For example, it may predict 0.9 for inputs where the accuracy is only 0.6. Notice that models with small test errors can still be overconfident, and therefore can benefit from label smoothing.
	
  * ### Formula of Label Smoothing
	Label smoothing replaces one-hot encoded label vector y_hot with a mixture of y_hot and the uniform distribution: 
      ```bash
    y_ls = (1 - α) * y_hot + α / K
      ```
	here K is the number of label classes, and α is a hyperparameter that determines the amount of smoothing. If α = 0, we obtain the original one-hot encoded y_hot. If α = 1, we get the uniform distribution.
	
  * ### Motivation of Label Smoothing
	Label smoothing is used when the loss function is cross entropy, and the model applies the softmax function to the penultimate layer’s logit vectors z to compute its output probabilities p. In this setting, the gradient of the cross entropy loss function with respect to the logits is simply


#### Grad Scaler
  * ### Introduction 
	16-bit precision may not always be enough for some computations. One particular case of interest is representing gradient values, a great portion of which are usually small values. Representing them with 16-bit floats often leads to buffer underflows (i.e. they’d be represented as zeros). This makes training neural networks very unstable. GradScalar is designed to resolve this issue. It takes as input your loss value and multiplies it by a large scalar, inflating gradient values, and therefore making them represnetable in 16-bit precision. It then scales them down during gradient update to ensure parameters are updated correctly. This is generally what GradScalar does. But under the hood GradScalar is a bit smarter than that. Inflating the gradients may actually result in overflows which is equally bad. So GradScalar actually monitors the gradient values and if it detects overflows it skips updates, scaling down the scalar factor according to a configurable schedule. (The default schedule usually works but we may need to adjust that for our use case.)

	Using GradScalar is very easy in practice:
	<image src='assets/gradscaler.png' >


	Note that we first create an instance of GradScalar. In training loop we call GradScalar.scale to scale the loss before calling backward to produce inflated gradients, we then use GradScalar.step which (may) update the model parameters. We then call GradScalar.update which performs the scalar update if needed. 

---

### Graphs

---

### Images

---

### References:

---
