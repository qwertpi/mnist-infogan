# mnist-infogan
A GAN loosely based on [INFOGAN](https://arxiv.org/abs/1606.03657) trained to conditionally generate MNIST images. This basicly just means that the discrimnator also attempts to predict class but is only trained to do this for the classes of the training data and the generator aims to fool the discriminator but have the class correctly preidcted.

WARNING: Only Linux is officaly supported  
Feedback and pull requests are very welcome

Inspired by https://github.com/eriklindernoren/Keras-GAN/blob/master/infogan/infogan.py  
![Example output](output.png?raw=true "Example output")  
<p align="center">
<img height=200, width=200, src="interpolated.gif?raw=True">
</p>

## Copyright
Copyright © 2019  Rory Sharp All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with this program.  If you have not received this, see <http://www.gnu.org/licenses/gpl-3.0.html>.

For a summary of the licence go to https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)

## Prerequisites
### For One Liner
* Curl `apt-get install curl`
### For Manual Install
* [Python 3](https://www.python.org/downloads/)
* Keras `pip3 install keras`
* Numpy `pip3 install numpy`
* TensorFlow `pip3 install tensorflow`
* h5py `pip3 install h5py`
* matplotlib `pip3 install matplotlib`
* tqdm (for training only) (likely to already be installed) `pip3 install tqdm`
* keract (for intermediate layer visualisation and linear interpolation only) `pip3 install keract`
* imageio (for linear interpolation only) `pip3 install imageio`
* libhdf5 (only needed on some systems) `sudo apt-get install libhdf5-serial-dev`

## One-liner install
`curl https://raw.githubusercontent.com/qwertpi/mnist-infogan/master/install.bash | bash`
## Usage
0\. Download this repo  
### Training (Optional)
1\. Create a folder called images that will be used to peridocly save generated imaes to during training  
2\. Change the `if True` statements for checkpoint loading to `if False` when running the code for the first time or it will load the pretrained generator in this repo and spoil your fun  
3\. Run train.py for as long as you want (I recomend around 40,000 epochs), I trained using google colabatory  
### Generating
4\. Run generate.py  
5\. Marvel at what modern neural networks can do  
### Intermideate layer visualistion
6\. If you run viz.py you will be able to see the outputs of the convolutional layers inbetween the input and output
### Linear interpolation
7\. Decide whether you want to be shown the images as a matplotlib plot or as a gif and change the value of save within interpolate.py accordingly  
8\. Run interpolate.py  
9\. Provide the starting and finishing classes that the model will smoothly move between (I found that it goes through imbetween classes)  
10\. If you opted to save find the interpolated.gif file  
