# StackGANs based Text to Image Generation

This repository provides stage-wise implementation of StackGANs to produce photo-realistic images from given text.

## Requirements

- `Python 3.9`
- `tensorflow 2.8`
- `numpy 1.22.3`

## Usage

- To train Stage-1 StackGAN : run `Train_Stage_1_GAN.py`
- To test Stage-2 StackGAN : run `Train_Stage_2_GAN.py`
- To see the ***Stage-1 StackGAN*** and ***Stage-2 StackGAN*** implementations, please check `Stage_1_GAN.py` and `Stage_2_GAN.py` respectively.
- All hyperparamters to control training and testing of ***StackGANs*** are provided in `Train_Stage_1_GAN.py` and `Train_Stage_2_GAN.py` files.

## Results

The eventual outcomes of both `Stage-1 StackGAN` and `Stage-2 StackGAN` can be seen against each given input text in the following attached image:-

![alt text](https://github.com/fork123aniket/Text-to-Image-Synthesis-using-StackGANs/tree/main/Images/Result.png)
