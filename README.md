# Super-Resolution Generative Adversarial Network (SRGAN) for License Plate Recognition
## Description
Master's Thesis research that utilizes SRGANs for license plate image quality improvement
and uses Google's Tesseract OCR for license plate recognition. 
Recognition accuracy was tested for both low-resolution and super-resolution images.

SRGAN utilized a large generated dataset of roughly 5000 images.
The images initially were generated with a size of 512x512 pixels and downsampled
to the various input sizes needed for training and validation. 
The input sizes, 16x16, 32x32, and 64x64 pixels, were fed into different 
instances of the SRGAN algorithm to produce the expected results.
In an effort to test the limitations, each input size was up-sampled in the SRGAN to produce multiple output image sizes.
The scales tested are specified in the following table:

[//]: # (Insert table here for the combinations)

Research consisted of:

- Generating the image dataset and splitting into training and testing subsets
- Downsampling the dataset into the input sizes during runtime
- Training and testing the SRGAN model
- Determining the accuracy of the model using labels generated during dataset creation.
