# Histopathology-Stain-Color-Normalization
Deep Convolutional Gaussian Mixture Model for Stain-Color Normalization in Histopathological H&amp;E Images. The TensorFlow implementation.

## Overview ##

Stain-color variation degrades the performance of the computer-aided diagnosis (CAD) systems. In the presence of severe color variarion between training set and test set in histopathological images, current CAD systems including deep learning models suffer from such an undesirable effect. Stain-color normalization is known as a remedy.

![alt text](https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Color_Normalization.png)

## Methodology ##
Stain-color normalization model can be defined as a generative models that by applying on input image can create different color copies of input image to somehow the converted image contain specific chromatic distribution. Our proposed method contains two stage: (1) Fitting a Gaussian mixture model (GMM) by considering the shape and apearance of image content structures. To do so the visual representation and modeling of convolutional neyral networks (CNNs) are exploited. (2) transforming the estimated distribution to any arbitary distribution that computed from a secondary (template) image.

## Features ##
* Fully unsupervised end-to-end learning algorithm
* Best performance in color constancy in normalized images
* Lack of any threshold and prioir assumption about image contents
* Considering image content structures in fitting color distribution to image pixels

<img  width="200" height="175" src=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Images/Example_GMM_Image.png> <img  width="200" height="175" src=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Images/Example_GMM_Classes.png> <img  width="200" height="175" src=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Images/Example_DCGMM_Classes.png>

**Tissue class membership computed by the CNN**; (red) nucleai; (green) surrounding tissues; (blue) background.

## Results ##

**Template image**

<img  width="200" height="175" src=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Images/Template.png> 

**Stain-color conversion**

<img  width="400" height="350" src=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Images/P1C7-L2di1Snap1.png> ==> <img  width="400" height="350" src=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Images/DCGMM_P1C7-L2di1Snap1.png>
<img  width="400" height="350" src=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Images/P2C25-L4di1Snap3.png> ==> <img  width="400" height="350" src=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Images/DCGMM_P2C25-L4di1Snap3.png>
<img  width="400" height="350" src=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Images/P3C4-L3ma1Snap3.png> ==> <img  width="400" height="350" src=https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/Images/DCGMM_P3C4-L3ma1Snap3.png>

Dataset
For evaluating the algorithm, <a href="https://camelyon17.grand-challenge.org/">CAMELYON17</a> dataset can be used.

## Citing DCGMM ##
If you find DCGMM useful in your research, please consider citing:

<a href="https://openreview.net/pdf?id=SkjdxkhoG">Zanjani, F. G., Zinger, S., Bejnordi, B. E., & van der Laak, J. A. (2018). Histopathology Stain-Color Normalization Using Deep Generative Models.</a>

## License ##
Stian-color normalization by using DCGMM is released under the free GNU license.

## Acknowledgement ##
This work is a part of 3DPathology project and has been funded by ITEA3 (Grant number: ITEA151003).


