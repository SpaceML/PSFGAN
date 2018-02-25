# PSFGAN

This project is the implementation of the paper "<tt>PSFGAN</tt>: a generative adversarial network system for separating quasar point sources and host galaxy light". This code can be used to create images for training sets and to apply pretrained models to new data.

## Prerequisites
Linux or OSX
NVIDIA GPU + CUDA CuDNN for training on GPU's

We further use the following tools to create training and testing sets:
<tt>GALFIT</tt>, <tt>Source Extractor</tt>

## Dependencies
Training requires the following python packages: <tt>tensorflow</tt>, <tt>numpy</tt>

Testing requires the following python packages: <tt>tensorflow</tt>, <tt>numpy</tt>, <tt>astropy</tt> (testing output is saved as a .fits file)

Creating training and testing sets requires the following python packages: <tt>numpy</tt>, <tt>scipy</tt>, <tt>matplotlib</tt>, <tt>astropy</tt>, <tt>photutils</tt>,

## Get our code
Clone this repo:
```bash
git clone https://github.com/SpaceML/PSFGAN.git
cd  PSFGAN/
```

## Run our code
First modify the file config.py. The relevant parameters are the following:
* run_case
* stretch_type
* scale_factor

### Create training/validation/testing sets

### Train PSFGAN

### Run a trained model