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

  One of the best things about STN is the ability to simply plug it into any existing CNN with very little modification.

  <b>Just like that 😉 </b>
  ```python
  # ==> Model <===
  # +--------------------------------------------------------------+
      def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

      localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
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
  

* ## <b>Vision Transformer Classes</b>
  * ### <b>Block Class</b>

    * ### Architechture
    * ### About the Class
  * ### <b>Embeddings </b>
    * ### Architechture
    * ### About the Class
  * ### <b>MLP</b>

    * ### Architechture
    * ### About the Class
  * ### <b>Attention</b>

    * ### Architechture
    * ### About the Class
  * ### <b>Encoder</b>

    * ### Architechture

    * ### About the Class
--- 
## Results



---
## Referances 
1. https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
2. 
