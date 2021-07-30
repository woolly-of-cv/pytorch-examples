# CIFAR10 with Spacial Transformer
### This file is submitted as part of Assignment 12 for EVA6 Course.
## Contributors

* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)
---
## Table of Contents
  - [About the Model and Architechture](#about-the-model)
  - [Result](#result)
  - [Referances](#referances)

---

## About the Model and Architechture   
* ## Introduction


  Spatial transformer networks are a generalization of differentiable attention to any spatial transformation. Spatial transformer networks (STN for short) allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model. For example, it can crop a region of interest, scale and correct the orientation of an image. It can be a useful mechanism because CNNs are not invariant to rotation and scale and more general affine transformations.

  One of the best things about STN is the ability to simply plug it into any existing CNN with very little modification. Spatial transformers can be incorporated into CNNs to benefit multifarious tasks:
  - Image classification: Suppose a CNN is trained to perform multi-way classification of images according to whether they contain a particular digit – where the position and size of the digit may vary significantly with each sample (and are uncorrelated with the class); a spatial transformer that crops out and scale normalizes the appropriate region can simplify the subsequent classification task, and lead to superior classification performance
  - Co-localisation: Given a set of images containing different instances of the same (but unknown) class, a spatial transformer can be used to localise them in each image
  - Spatial Attention: A key benefit of using attention is that transformed (and so attended), lower resolution inputs can be used in favour of higher resolution raw inputs, resulting in increased computational efficiency.
  
  ## Spatial Transformation Matrices:
  
  - Affine Transformation
  - Projective Transformation
  - Thin Plate Spline Transformation
  
  ## Spatial Transformers:
  
  This is a differentiable module which applies a spatial transformation to a feature map during a single forward pass, where the transformation is conditioned on the particular input, producing a single output feature map. For multi-channel inputs, the same warping is applied to each channel. The spatial transformer mechanism is split into three parts. 
  - In order of computation, first a localisation network takes the input feature map, and through a number of hidden layers outputs the parameters of the spatial transformation that should be applied to the feature map – this gives a transformation conditional on the input. 
  - Then, the predicted transformation parameters are used to create a sampling grid, which is a set of points where the input map should be sampled to produce the transformed output. This is done by the grid generator
  - Finally, the feature map and the sampling grid are taken as inputs to the sampler, producing the output map sampled from the input at the grid points 

  ```python
  # ==> Model <===
  # +--------------------------------------------------------------+
         self.localization = nn.Sequential(
         nn.Conv2d(3, 64, kernel_size=7),
         nn.MaxPool2d(2, stride=2),
         nn.ReLU(True),
         nn.Conv2d(64, 128, kernel_size=5),
         nn.MaxPool2d(2, stride=2),
         nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size(1) * xs.size(2) * xs.size(3))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x
  # +--------------------------------------------------------------+

  # ==> Visualizing the STN Results <===
  # +--------------------------------------------------------------+
  def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

  for epoch in range(1, 20 + 1):
      train(epoch)
      test()
  # Visualize the STN transformation on some input batch
  visualize_stn()

  plt.ioff()
  plt.show()

  # +--------------------------------------------------------------+
  ```
* ## Depicting Spatial Transformer Networks
  Spatial transformer networks boils down to three main components :
  * The localization network is a regular CNN which regresses the transformation parameters. The transformation is never learned explicitly from this dataset, instead the network learns automatically the spatial transformations that enhances the global accuracy.
  * The grid generator generates a grid of coordinates in the input image corresponding to each pixel from the output image.
  * The sampler uses the parameters of the transformation and applies it to the input image.

  $~~~~~~~~~~~$
  <p align="center">
    <img src='assets/spatial_transformer.png'>
  </p>
  

--- 
## Results



---
## Referances 
1. https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
2. 
