# Morphology-Aware Interactive Keypoint Estimation (MICCAI 2022) - Official PyTorch Implementation

[<ins>__[Paper]__</ins>](https://arxiv.org/abs/2209.07163) &nbsp; 
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


### Preparing code and model files
1. Create ``code``, ``pretrained_models``, and ``save`` folders.
    ```
    mkdir code
    mkdir code/pretrained_models
    mkdir save
    ```
2. Clone this repository in the ``code`` folder:
    ```
    cd code
    git clone https://github.com/seharanul17/interactive_keypoint_estimation
    ```

3. To train our model based on the pre-trained HRNet model, download its model file from [here](https://github.com/HRNet/HRNet-Image-Classification). 
Place the downloaded file in the ``pretrained_models`` folder. Related code line can be found [here](https://github.com/seharanul17/interactive_keypoint_estimation/blob/7f50ec271b9ae9613c839533d3958110405d04f5/model/iterativeRefinementModels/RITM_SE_HRNet32.py#L29).
   

4. To test our pre-trained model, download the folder containing our model file and config file from [here](https://www.dropbox.com/sh/m53iqw9loddqhfq/AAD0KuCCxpXsBE435Hw3KJU8a?dl=0).
Place the downloaded folder into the ``save`` folder.
Related code line can be found [here](https://github.com/seharanul17/interactive_keypoint_estimation/blob/7f50ec271b9ae9613c839533d3958110405d04f5/util.py#L77).



### Preparing data

We provide the code to conduct experiments on a public dataset, the AASCE challenge dataset.

1. Download the data.
    - The AASCE challenge dataset can be obtained from [SpineWeb](http://spineweb.digitalimaginggroup.ca/index.php?n=main.datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images). 
    - The AASCE challenge dataset corresponds to `Dataset 16: 609 spinal anterior-posterior x-ray images' on the webpage.
    
2. Preprocess the downloaded data.
    - Set the variable ``source_path `` in the ``data_preprocessing.py`` file as your dataset path.
    - Run the following command: 
        ```
        python data_preprocessing.py
        ```
        

### Usage
- To run the training code, run the following command:
    ```
    bash train.sh 
    ```
- To test the pre-trained model: 
   1. Locate the pre-trained model in the ``../save/`` folder.
   2. Run the test code:
        ```
        bash test.sh
        ```
- To test your own model:
   1. Change the value of the argument ``--only_test_version {your_model_name}`` in the ``test.sh`` file.
   2. Run the test code:
        ```
        bash test.sh
        ```

When the evaluation ends, the mean radial error (MRE) of model prediction and manual revision will be reported.
The ``sargmax_mm_MRE`` corresponds to the MRE reported in Fig. 4.


## Results
The following table compares the refinement performance of our proposed interactive model and manual revision.
Both models revise the same initial prediction results of our model. The number of user modifications is prolonged from zero (initial prediction) to five.
The model performance is measured using mean radial errors on the AASCE dataset.
For more information, please see Fig. 4 in our main manuscript.

- "Ours (model revision)" indicates automatically revised results by the proposed interactive keypoint estimation approach.
- "Ours (manual revision)" indicates fully-manually revised results by a user without the assistance of an interactive model.

|      Method     	| No. of user modification | | |  | | |
|:----------------:	|:-----------------------:	|:-------:	|:-------:	|:-------:	|:-------:	|:-------:	|
|              	| 0 (initial prediction)	| 1	| 2	| 3  	| 4	| 5	|
| Ours (model revision) |          58.58          	|  35.39  	|  29.35  	|  24.02  	|  21.06  	|  17.67  	|
|  Ours (manual revision) 	|          58.58          	|  55.85  	|  53.33  	|  50.90  	|  48.55  	|  47.03  	|


## Citation

If you find this work or code is helpful in your research, please cite:
```
@inproceedings{kim2022morphology,
  title={Morphology-Aware Interactive Keypoint Estimation},
  author={Kim, Jinhee and 
          Kim, Taesung and 
          Kim, Taewoo and 
          Choo, Jaegul and 
          Kim, Dong-Wook and 
          Ahn, Byungduk and 
          Song, In-Seok and 
          Kim, Yoon-Ji},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={675--685},
  year={2022},
  organization={Springer}
}
```

