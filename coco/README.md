# Analyzing a COCO data format

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


To get insites about this dataset we will start analyzing it by first loading dataset. 

We will be using `pandas` to do our analysis.

### References

* https://gist.github.com/jinyu121/e530dc9767d8f83c08f3582c71a5cbc8