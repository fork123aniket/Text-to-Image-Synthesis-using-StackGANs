# StackGANs-based Text-to-Image Generation

This repository provides Stage-wise implementation of StackGANs to produce photo-realistic images from given text. The Stage-1 GAN sketches the primitive shape and colors of the object based on the given text description, yielding Stage-1 low-resolution images. On the other hand, The Stage-2 GAN takes Stage-1 results and text descriptions as inputs and generates high-resolution images with photo-realistic details. Moreover, it is able to rectify defects in Stage-1 results and add compelling details with the refinement process.

## Requirements

- `Python`
- `tensorflow`
- `keras`
- `numpy`
- `pandas`
- `PIL`
- `matplotlib`

## Usage

### Data
The training of StackGAN has been performed on CUB dataset. CUB contains 200 bird species with 11,788 images. Since 80% of birds in this dataset have object-image size ratios of less than 0.5, as a pre-processing step, cropping has been executed for all images to ensure that bounding boxes of birds have greater-than-0.75 object-image size ratios. The dataset can either be downloaded from [***here***](https://drive.google.com/open?id=0B3y_msrWZaXLT1BZdVdycDY5TEE) or can be obtained by running `wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz` command.

### Training and Testing
- To train ***Stage-1 StackGAN*** : run `Train_Stage_1_GAN.py`
- To test ***Stage-2 StackGAN*** : run `Train_Stage_2_GAN.py`
- To see the ***Stage-1 StackGAN*** and ***Stage-2 StackGAN*** implementations, please check `Stage_1_GAN.py` and `Stage_2_GAN.py` respectively.
- All hyperparamters to control training and testing of ***StackGANs*** are provided in `Train_Stage_1_GAN.py` and `Train_Stage_2_GAN.py` files.

## Results

The eventual outcomes of both ***Stage-1 StackGAN*** and ***Stage-2 StackGAN*** can be seen against each given input text in the following attached image:-

![alt text](https://github.com/fork123aniket/Text-to-Image-Synthesis-using-StackGANs/blob/main/Images/Result.png)
