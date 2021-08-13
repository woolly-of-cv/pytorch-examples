# End-to-End Object Detection with Transformers with Custom Dataset

### This file is submitted as part of Assignment 14 for EVA6 Course.
---
## Contributors

* [Ammar Adil](https://github.com/adilsammar)
* [Krithiga](https://github.com/BottleSpink)
* [Shashwat Dhanraaj](https://github.com/sdhanraaj12)
* [Srikanth Kandarp](https://github.com/Srikanth-Kandarp)
---
## Table of Contents
  * [Introduction](#introduction)
    + [Encoder-Decoder Architecture](#encoder-decoder-architecture)
    + [What is Bipartite Loss and why ?](#what-is-bipartite-loss-and-why--)
    + [What are Object Queries ?](#what-are-object-queries--)
  * [References](#references)
---
 ## Introduction
The goal of object detection is to predict a set of bounding boxes and category labels for each object of interest. Modern detectors address this set prediction task in an indirect way, by defining surrogate regression and classification prob- lems on a large set of proposals ,anchors, or window centers. Their performances are significantly influenced by postprocessing steps to col- lapse near-duplicate predictions, by the design of the anchor sets and by the heuristics that assign target boxes to anchors. To simplify these pipelines, we propose a direct set prediction approach to bypass the surrogate tasks. This end-to-end philosophy has led to significant advances in complex structured pre- diction tasks such as machine translation or speech recognition, but not yet in object detection: previous attempts either add other forms of prior knowledge, or have not proven to be competitive with strong baselines on chal- lenging benchmarks. This paper aims to bridge this gap.

  <p align="center"> 
  <image src='assets/DETR_1.png'>
  </p>
    

We streamline the training pipeline by viewing object detection as a direct set prediction problem. We adopt an encoder-decoder architecture based on trans- formers, a popular architecture for sequence prediction. The self-attention mechanisms of transformers, which explicitly model all pairwise interactions be- tween elements in a sequence, make these architectures particularly suitable for specific constraints of set prediction such as removing duplicate predictions.
Our Detection Transformer predicts all objects at once, and is trained end-to-end with a set loss function which performs bipar- tite matching between predicted and ground-truth objects. DETR simplifies the detection pipeline by dropping multiple hand-designed components that encode prior knowledge, like spatial anchors or non-maximal suppression. Unlike most existing detection methods, DETR doesn’t require any customized layers, and thus can be reproduced easily in any framework that contains standard CNN and transformer classes.1.
Compared to most previous work on direct set prediction, the main features of DETR are the conjunction of the bipartite matching loss and transformers with (non-autoregressive) parallel decoding. In contrast, previous work focused on autoregressive decoding with RNNs. Our matching loss function uniquely assigns a prediction to a ground truth object, and is invariant to a permutation of predicted objects, so we can emit them in parallel.
We evaluate DETR on one of the most popular object detection datasets, COCO, against a very competitive Faster R-CNN baseline. Faster R- CNN has undergone many design iterations and its performance was greatly improved since the original publication. Our experiments show that our new model achieves comparable performances. More precisely, DETR demonstrates significantly better performance on large objects, a result likely enabled by the non-local computations of the transformer. It obtains, however, lower perfor- mances on small objects. We expect that future work will improve this aspect in the same way the development of FPN did for Faster R-CNN.
  * ###  Encoder-Decoder Architecture

  <p align="center"> 
  <image src='assets/e&d_arch.png'>
  </p>

    Image features from the CNN backbone are passed through the transformer encoder, together with spatial positional encoding that are added to queries and keys at every multi- head self-attention layer. Then, the decoder receives queries (initially set to zero), output positional encoding (object queries), and encoder memory, and produces the final set of predicted class labels and bounding boxes through multiple multi- head self-attention and decoder-encoder attention. The first self-attention layer in the first decoder layer can be skipped.

    * ### Computational Complexity 
      Every self-attention in the encoder has complex- ity ```O(d2HW +d(HW )2): O(d′d)``` is the cost of computing a single query/key/value embeddings ```(and Md′ = d)```, while ```O(d′(HW)2)``` is the cost of computing the at- tention weights for one head. Other computations are negligible. In the decoder, each self-attention is in ```O(d2N +dN2)```, and cross-attention between encoder and decoder is in ```O(d2(N +HW)+dNHW)```, which is much lower than the encoder since N ≪ HW in practice.

    * ### FLOPS Computation
     Given that the FLOPS for Faster R-CNN depends on the number of proposals in the image, we report the average number of FLOPS for the first 100 images in the COCO 2017 validation set. We compute the FLOPS with the tool flop count operators from Detectron2. We use it without modifications for Detectron2 models, and extend it to take batch matrix multiply (bmm) into account for DETR models.
  * ### What is Bipartite Loss and why ? 
    In bipartite matching loss what we actually do is compare the predicted classes + bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N (so if an image only contains 4 objects, 96 annotations will just have a “no object” as class and “no bounding box” as bounding box). The Hungarian matching algorithm is used to find an optimal one-to-one mapping of each of the N queries to each of the N annotations. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and generalized IoU loss (for the bounding boxes) are used to optimize the parameters of the model.

  * ### What are Object Queries ? 
    After the hidden states are passed through the encoder then object queries are sent through the decoder. This is a tensor of shape (batch_size, num_queries, d_model), with num_queries typically set to 100 and initialized with zeros. These input embeddings are learnt positional encodings that the authors refer to as object queries, and similarly to the encoder, they are added to the input of each attention layer. Each object query will look for a particular object in the image. The decoder updates these embeddings through multiple self-attention and encoder-decoder attention layers to output decoder_hidden_states of the same shape: (batch_size, num_queries, d_model). Next, two heads are added on top for object detection: a linear layer for classifying each object query into one of the objects or “no object”, and a MLP to predict bounding boxes for each query.
  * ### Understanding DETR with the code 



---
## Referances 

1. https://opensourcelibs.com/lib/finetune-detr
2. http://www.cs.toronto.edu/~sven/Papers/bipartite.pdf
3. https://research.fb.com/wp-content/uploads/2020/08/End-to-End-Object-Detection-with-Transformers.pdf
4. https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46?gi=fc59354d7ac5
  