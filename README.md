# Morphology-Aware Interactive Keypoint Estimation (MICCAI 2022) - Official PyTorch Implementation

[<ins>__[Paper]__</ins>]() &nbsp; 
&nbsp; 
 [<ins>__[Project page]__</ins>](https://seharanul17.github.io/interactive_keypoint_estimation/)
&nbsp;  &nbsp; 
 [<ins>__[Video]__</ins>](https://youtu.be/Z5gtLviQ_TU)

## Introduction
This is the official Pytorch implementation of [Morphology-Aware Interactive Keypoint Estimation]().

Diagnosis based on medical images, such as X-ray images, often involves manual annotation of anatomical keypoints. However, this process involves significant human efforts and can thus be a bottleneck in the diagnostic process. To fully automate this procedure, deep-learning-based methods have been widely proposed and have achieved high performance in detecting keypoints in medical images. However, these methods still have clinical limitations: accuracy cannot be guaranteed for all cases, and it is necessary for doctors to double-check all predictions of models. In response, we propose a novel deep neural network that, given an X-ray image, automatically detects and refines the anatomical keypoints through a user-interactive system in which doctors can fix mispredicted keypoints with fewer clicks than needed during manual revision. Using our own collected data and the publicly available AASCE dataset, we demonstrate the effectiveness of the proposed method in reducing the annotation costs via extensive quantitative and qualitative results.

## Environment
The code was developed using python 3.8 on Ubuntu 18.04.

The experiments were performed on a single GeForce RTX 3090 in the training and evaluation phases.

##  Quick start 

### Prerequisites
Install following dependencies:
- Python 3.8
- torch == 1.8.0
- albumentations == 1.1.0
- munch
- tensorboard
- pytz
- tqdm


### Code preparation
1. Create ``code`` and ``save`` folders.
    ```
    mkdir code
    mkdir save
    ```
2. Clone this repository in the ``code`` folder:
    ```
    cd code
    git clone https://github.com/seharanul17/interactive_keypoint_estimation
    ```

### Dataset preparation

We provide the code to conduct experiments on a public dataset, the AASCE challenge dataset.

1. Prepare the data.
    - The AASCE challenge dataset can be obtained from [SpineWeb](http://spineweb.digitalimaginggroup.ca/index.php?n=main.datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images). 
    - The AASCE challenge dataset corresponds to `Dataset 16: 609 spinal anterior-posterior x-ray images' on the webpage.
    
2. Preprocess the data.
    - Set the variable ``source_path `` in the ``data_preprocessing.py`` file as your dataset path.
    - Run the ``data_preprocessing.py`` file: 
    
    ```python data_preprocessing.py```

### Download the pre-trained model

To test the pre-trained model, download the model file from [here](https://www.dropbox.com/sh/m53iqw9loddqhfq/AAD0KuCCxpXsBE435Hw3KJU8a?dl=0).

### Usage
#### Run training code
Run the following command:

    ```bash train.sh ```
#### Run evaluation code 
    - To test the pre-trained model: 
        1. Locate the pre-trained model in the ``../save/`` folder.
        2. Run the test code:
        
        ```bash test.sh```
    - To test your own model:
        1. Change the value of the argument ``--only_test_version {your_model_name}`` in the ``test.sh`` file.
        2. Run the test code:
        
        
        ```bash test.sh```

When the evaluation ends, the mean radial error (MRE) of model prediction and manual revision will be reported.
The ``sargmax_mm_MRE`` corresponds to the MRE reported in Fig. 4.


## Results
The following table compares the refinement performance of our proposed interactive model and manual revision.
Both models revise the same initial prediction results of our model. The number of user modifications is prolonged from zero (initial prediction) to five.
The model performance is measured using mean radial errors on the AASCE dataset.
For more information, please see Fig. 4 in our main manuscript.

- "Ours (model revision)" indicates automatically revised results by the proposed interactive keypoint estimation approach.
- "Ours (manual revision)" indicates fully-manually revised results by a user without the assistance of an interactive model.

|      Method     	| No. user modification |||||	|
|                  	| 0	| 1	| 2	| 3  	| 4	| 5	|
|:----------------:	|:-----------------------:	|:-------:	|:-------:	|:-------:	|:-------:	|:-------:	|
| Ours (model revision) |          58.58          	|  35.39  	|  29.35  	|  24.02  	|  21.06  	|  17.67  	|
|  Ours (manual revision) 	|          58.58          	|  55.85  	|  53.33  	|  50.90  	|  48.55  	|  47.03  	|

