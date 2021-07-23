# Analyzing COCO data format

## Contributors

* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)
---

## Table of Contents
  - [Coco Dataset](#tiny-imagenet-dataset)
  - [Loading the Dataset](#loading-the-dataset)
  - [Calculation](#calculation)
      - [Class Spread](#class-spread)
  - [Anchor box](#anchor-box)
  - [Intersection over Union](#intersection-over-union)
  - [k-means](#k-means)
  - [References](#references)
  
  
## Coco Dataset:

We will be using a sample subset of data [link](./dataset/sample_coco.txt)

There are basically two data formats for bounding box

1. COCO Bounding box: (x-top left, y-top left, width, height)
2. Pascal VOC Bounding box :(x-top left, y-top left,x-bottom right, y-bottom right)

One we are concerned in this article is COCO format.

Subset Dataset we are using for analysis, each row of this file looks as below where,

    id: 0, height: 330, width: 1093, bbox:[69, 464, 312, 175],

    id - Image Id
    height - Image Original Height
    width - Image Original Width
    bbox - Bounding Box in COCO format (x-top left, y-top left, width, height)

## Loading the Dataset:

To get insights about this dataset we will start analyzing it by first loading the dataset. We will be using `pandas` to do our analysis. We read the data from sample_coco file and dump it as json file. From bbox, we derive x-centroid, y-centroid for bounding box, bounding box width and height. We populate the dataframe with the above columns and the classes.

## Calculation:

### Class Spread:

With pandas, the value counts for each class is then viewed. The below chart shows the same:

## Anchor box:

Object detection algorithms usually sample a large number of regions in the input image, determine whether these regions contain objects of interest, and adjust the boundaries of the regions so as to predict the ground-truth bounding boxes of the objects more accurately. Different models may adopt different region sampling schemes. Here we introduce one of such methods: it generates multiple bounding boxes with varying scales and aspect ratios centered on each pixel. These bounding boxes are called anchor boxes.

Suppose that the input image has a height of ℎ and width of 𝑤 . We generate anchor boxes with different shapes centered on each pixel of the image. when the center position is given, an anchor box with known width and height is determined. We can see that the shape of the returned anchor box variable Y is (batch size, number of anchor boxes, 4). After changing the shape of the anchor box variable Y to (image height, image width, number of anchor boxes centered on the same pixel, 4), we can obtain all the anchor boxes centered on a specified pixel position. In the following, we access the first anchor box centered on (250, 250). It has four elements: the  (𝑥,𝑦) -axis coordinates at the upper-left corner and the  (𝑥,𝑦) -axis coordinates at the lower-right corner of the anchor box. The coordinate values of both axes are divided by the width and height of the image, respectively; thus, the range is between 0 and 1. As we just saw, the coordinate values of the  𝑥  and  𝑦  axes in the variable boxes have been divided by the width and height of the image, respectively. When drawing anchor boxes, we need to restore their original coordinate values; thus, we define variable bbox_scale below.

## Intersection over Union:

An anchor box “well” surrounds the object in the image. If the ground-truth bounding box of the object is known, how can “well” here be quantified? Intuitively, we can measure the similarity between the anchor box and the ground-truth bounding box. The index is the size of their intersection divided by the size of their union.
 
In fact, we can consider the pixel area of any bounding box as a set of pixels. In this way, we can measure the similarity of the two bounding boxes by the index of their pixel sets. For two bounding boxes, we usually refer their index as intersection over union (IoU), which is the ratio of their intersection area to their union area. The range of an IoU is between 0 and 1: 0 means that two bounding boxes do not overlap at all, while 1 indicates that the two bounding boxes are equal.

## K-Means:

K -means clustering algorithm is very famous algorithm in data science. This algorithm aims to partition n observation to k clusters. Mainly it includes :

Initialization : K means (i.e centroid) are generated at random.
Assignment : Clustering formation by associating the each observation with nearest centroid.

Updating Cluster : Centroid of a newly created cluster becomes mean.

Assignment and Update are repitatively occurs untill convergence. The final result is that the sum of squared errors is minimized between points and their respective centroids.

### Why use K means?

K-means is computationally faster and more efficient compare to other unsupervised learning algorithms. Don't forget time complexity is linear.
It produces a higher cluster then the hierarchical clustering. More number of cluster helps to get more accurate end result.
Instance can change cluster (move to another cluster) when centroid are re-computed.
Works well even if some of your assumption are broken.

### What it really does in determining anchor box?

It will create a thouasands of anchor box (i.e Clusters in k-means) for each predictor that represent shape, location, size etc.
For each anchor box, calculate which object’s bounding box has the highest overlap divided by non-overlap. This is called Intersection Over Union or IOU.
If the highest IOU is greater than 50% ( This can be customized), tell the anchor box that it should detect the object it has highest IOU.
Otherwise if the IOU is greater than 40%, tell the neural network that the true detection is ambiguous and not to learn from that example.
If the highest IOU is less than 40%, then it should predict that there is no object.


## References: 

* https://gist.github.com/jinyu121/e530dc9767d8f83c08f3582c71a5cbc8
