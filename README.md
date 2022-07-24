# Image Inpainting

This is my implementation of [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723) with slight differences.

## Results

Here are the results of training for about 2 epochs on ImageNet.

Draw your mask and test the model by yourself:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sankek/ImageInpainting/blob/master/Drawn_Mask_Image_Inpainting.ipynb)

![Inpainting example](/figures/example1.png)
![Inpainting example](/figures/example2.png)
![Inpainting example](/figures/example3.png)

## Training 

The model was trained on [ImageNet](https://www.image-net.org/) dataset on 256x256 images.

For training I generate irregular masks with random walk and they look like this:

![Mask examples](/figures/mask_examples.png)

Training notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sankek/ImageInpainting/blob/master/training.ipynb)



