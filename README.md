





# Report for ECCV DIRA Data Challenge

## Introduction

Illustrations, drawings, technical diagrams and diagrams can help people quickly understand difficult concepts. This kind of visual information is very popular in our daily life. However, there are many challenges in processing such visual data. As shown in Fig 1., these images are similar as hand-drawn sketches, without background or other contextual information. These images are often presented in several different viewpoints, which increases the difficulty of image retrieval. This challenge requires participants to extract robust image representations from DeepPatent[1] dataset for abstract diagram retrieval. Given a query image, participants were required to retrieve images that belong to the same patent or representing similar objects. 

We found the patent retrieval task very similar to person re-identification task. Therefore, we design a framework for patent retrieval based on person Re-ID methods. The rest of our report is organized as follows: In Section 2, we describe our approach, including the overview, data augmentation, backbone network, loss function, post-processing, training details and dataset usage; In Section 3, we describe our code guidance.

<img src=".assets/image-20221009145459227.png" alt="image-20221009145459227" style="zoom:50%;" />

<center style="font-size:16px;color:#000">Fig 1.</center> 

## Method

### Overview

The framework of our method is shown in Fig 2.

![image-20221009162405174](.assets/image-20221009162405174.png)

<center style="font-size:16px;color:#000">Fig 2.</center> 

### Data Augmentation

We applied autoaugment method[1], which can search appropriate composed augmentation including flip, rotate, bright/color change,etc. Random Erasing is also utilize to avoid model overfitting.

### Backbone Network

We adopt ResNet101[2] as our backbone. IBN-Net[3] and Non-local[4] modules are added to ResNet101 to get more robust features.  Both modules are used to help model to learn a better universal image representations. The input image size is set to 256 in the first stage and 384 in the second stage.  A BN layer and Generalized Mean Pooling (GEM) are applied at the end of backbone network to extract retrieval features. Finally, a classifier layer is used to output the probability of different IDs.

### Testing

In the test phase, the test images are resized to  384×384 resolution and inputted into the model to get feature representation. We first calculate the Euclidean distance between the query and gallery features. Then Query expansion(QE) is used for post-processing. Given a query image and its top k similar gallery images. The new query feature is constructed by averaging the verified gallery features and the query feature.  The new query feature is used for following image retrieve to boost performance.
### Loss Functions

In the training phase,  circle loss[5] and soft-margin triplet loss[6] are utilized for metric learning task.

### Training

We divide our training procedure into two stages . In the first stage, we train our network on 224×224 images for the first 70 epochs. and then finetune the network at 384×384 resolution for 50 epochs.  The learning rate in stage 1 is set to 0.000035 initially and gradually increased to 0.00035 during the 2k iteration by linear warm up. Then the learning rate starts from 0.00035 and decays to $7.7 \times 10^{-7}$ by cosine rule after 9k iterations.  The learning rate in stage 2 is set to 0.000035 initially and decays to $7.7 \times 10^{-7}$ by cosine rule. Random Erasing probability is set to 0.5. The optimizer is Adam. The weight decay is 0.0005.  Each min-batch includes 64 images, which includes 16 objects and 4 images per object.

### Dataset

We only used the DeepPatent dataset for training and validation. 

### Reference

[1] Kucer M, Oyen D, Castorena J, et al. DeepPatent: Large scale patent drawing recognition and retrieval[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2022: 2309-2318.

[2] Cubuk E D, Zoph B, Mane D, et al. Autoaugment: Learning augmentation policies from data[J]. arXiv preprint arXiv:1805.09501, 2018.

We only utilize the DeepPatent dataset for training and validation, which provides more than 350,000 design patent drawings for the purpose of image retrieval.
[3] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

[4] Pan X, Luo P, Shi J, et al. Two at once: Enhancing learning and generalization capacities via ibn-net[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 464-479.

[5] Wang X, Girshick R, Gupta A, et al. Non-local neural networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7794-7803.

[6] Sun Y, Cheng C, Zhang Y, et al. Circle loss: A unified perspective of pair similarity optimization[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 6398-6407.

[7] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 815-823.





## Code Guidance for ECCV DIRA Data Challenge

The following is our code guidance  for DIRA Data Challenge

### Requirements

- Linux or macOS with python ≥ 3.6
- PyTorch ≥ 1.6
- [yacs](https://github.com/rbgirshick/yacs)
- Cython (optional to compile evaluation code)
- tensorboard (needed for visualization): `pip install tensorboard`
- gdown (for automatically downloading pre-train model)
- sklearn
- termcolor
- tabulate
- [faiss](https://github.com/facebookresearch/faiss) `pip install faiss-gpu`

For conda

```
conda create -n fastreid python=3.8
conda activate fastreid
conda install pytorch==1.7.1 torchvision tensorboard -c pytorch
pip install -r docs/requirements.txt
```

We use GPU 3090 for training and testing. The cuda version is 11.1, torch version is 1.7.1, the python version is 3.8.8.

### Dataset

Download the competition datasets patent data and codalab test set, and then unzip them under the datasets directory like: 

```
datasets
├── train_data
|	└── patent_data
|		└── I20180102
|		└── ...
|	└── train_patent_val.txt
|	└── train_patent_trn.txt
├── test_data
|	└── codalab_test_set
|	└── database
|	└── lists
|	└── queries

```

### Prepare Pre-trained Models

You can download the pre-trained model form this link: https://drive.google.com/drive/folders/1DMcAnAHZ54QZPi9KxkgPA9er_uE76dP3?usp=sharing. Then you should save it under the path of logs. The file tree should be like as:

```
logs
└── Patent	
    └── R101_384
    	└── model_best.pth
```

### Test

You can get the final result.npy by running:

```
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py --config-file ./configs/patent/sbs_R101-ibn_patent_test.yml --eval-only  MODEL.DEVICE "cuda:0"
```

It will generate result.npy  in the root dir, which is the final result. The test process takes approximately 4 hours.

### Training

```
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/patent/sbs_R101-ibn_patent_256.yml MODEL.DEVICE "cuda:0" 
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/patent/sbs_R101-ibn_patent_384.yml MODEL.DEVICE "cuda:0" 
```

We train our model through two stages. Stage1 train the original dataset at 256 $\times$ 256 resolution . Stage2 finetune the trainset at 384 $\times$ 384 resolution which is inspired by **[kaggle-landmark-2021-1st-place](https://github.com/ChristofHenkel/kaggle-landmark-2021-1st-place)**.  

