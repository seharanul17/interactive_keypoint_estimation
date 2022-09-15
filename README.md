# Morphology-Aware Interactive Keypoint Estimation (MICCAI 2022 - Paper ID 2100)

[<ins>__[Paper]__</ins>]() &nbsp; 
&nbsp; 
 [<ins>__[Project page]__</ins>](https://seharanul17.github.io/interactive_keypoint_estimation/)
&nbsp; 
 [<ins>__[Video]__</ins>](https://youtu.be/Z5gtLviQ_TU)

## Introduction
This is the official Pytorch implementation of [Morphology-Aware Interactive Keypoint Estimation]().

Diagnosis based on medical images, such as X-ray images, often involves manual annotation of anatomical keypoints. However, this process involves significant human efforts and can thus be a bottleneck in the diagnostic process.
To fully automate this procedure, deep-learning-based methods have been widely proposed and have achieved high performance in detecting keypoints in medical images. 
However, these methods still have clinical limitations: accuracy cannot be guaranteed for all cases, and it is necessary for doctors to double-check all predictions of models. In response, we propose a novel deep neural network that, given an X-ray image, automatically detects and refines the anatomical keypoints through a user-interactive system in which doctors can fix mispredicted keypoints with fewer clicks than needed during manual revision.
Using our own collected data and the publicly available AASCE dataset, we demonstrate the effectiveness of the proposed method in reducing the annotation costs via extensive quantitative and qualitative results. A demo video of our approach is available on our [project webpage](https://seharanul17.github.io/interactive_keypoint_estimation/).


## Prerequisites
Install following dependencies:
- Python 3.8
- torch == 1.8.0
- albumentations == 1.1.0
- munch
- tensorboard
- pytz
- tqdm


## Dataset

Due to the privacy problem, we can not make the Cephalometric X-ray dataset public.
Instead, we provide the code for experiments on the AASCE challenge dataset.

#### Dataset preparation
1. Prepare the dataset.
    - The dataset can be obtained from the link below. The AASCE dataset is Dataset 16 in the linked page.
    - link: http://spineweb.digitalimaginggroup.ca/index.php?n=main.datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images

2. Preprocess the dataset.
    - Change the``source_path `` in ``data_preprocessing.py`` to your AASCE challenge dataset path.
    - Run the command ``python data_preprocessing.py``

## Pre-trained model

Since the file size of the pre-trained model is over 100MB, we can not upload the model on the GitHub page.
Therefore, to test the pre-trained model, you should download the model file from this [link](https://www.dropbox.com/sh/m53iqw9loddqhfq/AAD0KuCCxpXsBE435Hw3KJU8a?dl=0).

## Usage
1. To train the model,
    - run ``bash train.sh ``
2. To test the model,
    - when evaluating the pre-trained model, 
        - download pre-trained model from the provided link and locate the files at ``../save/`` folder.
        - Then, run ``bash test.sh``
    - when evaluating your own model,
        - change the argument ``--only_test_version {saved model folder name}`` in ``test.sh``
        - and run ``bash test.sh``

The results are mean radial error (MRE) of model prediction and manual revision.
The ``sargmax_mm_MRE`` indicates the MRE reported in Fig. 4.

## Results
The table below shows the mean radial error (MRE) of our proposed model and manual revision from the initial prediction of the model on AASCE dataset, prolonging the number of interactions.
The values of the table are the same with Fig. 4.

- "Ours (model revision)" indicates the model prediction results of the proposed model. 
- "Ours (manual reivision)" indicates the manually revised results from the initial prediction of "Ours".

|      Method      	| MRE with 0 interactions 	| with 1  	| with 2  	| with 3  	| with 4  	| with 5  	|
|:----------------:	|:-----------------------:	|:-------:	|:-------:	|:-------:	|:-------:	|:-------:	|
| Ours (model revision) |          58.58          	|  35.39  	|  29.35  	|  24.02  	|  21.06  	|  17.67  	|
|  Ours (manual revision) 	|          58.58          	|  55.85  	|  53.33  	|  50.90  	|  48.55  	|  47.03  	|

