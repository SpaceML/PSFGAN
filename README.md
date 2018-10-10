# PSFGAN

This project is the implementation of the paper "<tt>PSFGAN</tt>: a generative adversarial network system for separating quasar point sources and host galaxy light". This code can be used to create images for training sets and to apply pretrained models to new data.

## Prerequisites
Linux or OSX

NVIDIA GPU + CUDA CuDNN for training on GPU's

We further use the following tools to create training and testing sets:
<tt>GALFIT</tt>, <tt>Source Extractor</tt>, SDSS PSF tool ("Stand Alone Code" downloaded from http://www.sdss.org/dr12/algorithms/read_psf/)

## Dependencies
The code currently runs on <tt>python 2.7</tt> but we are working on making it compatible for new versions of python.

Training requires the following python packages: <tt>tensorflow</tt>, <tt>numpy</tt>

Testing requires the following python packages: <tt>tensorflow</tt>, <tt>numpy</tt>, <tt>astropy</tt> (testing output is saved as a .fits file)

Creating training and testing sets requires the following python packages: <tt>numpy</tt>, <tt>scipy</tt>, <tt>matplotlib</tt>, <tt>astropy</tt>, <tt>photutils</tt>

## Get our code
Clone this repo:
```bash
git clone https://github.com/SpaceML/PSFGAN.git
cd  PSFGAN/
```

## Run our code

### Create training/validation/testing sets
*If you already have training/testing data or you are applying PSFGAN to real data, you can skip everything described in this section. If you don't have any data but you want to test our method, you can get the data from this Google Drive folder*

https://drive.google.com/open?id=15RNVBdCsonm2EQTxlEeEKO_zf2P-F6cN 

This will get you all the precomputed data necessary for a first training run. It is the z=0.1 dataset we describe in the paper.

*If you have original images and want to create a training/validation/testing set, you can perform the following steps.*
The code assumes a directory structure like the following.
```bash
core_path/
├── config.py
├── data.py
├── galfit.py
├── model.py
├── normalizing.py
├── photometry.py
├── roou.py
├── test.py
├── train.py
├── utils.py
└── z_0.1
    └── r-band
        ├── catalog_z_0.045_0.055.csv
        ├── fits_eval
        ├── fits_test
        └── fits_train
```
In this example there are two subfolders containing training, testing and validation images and a catalog. For SDSS data the catalog must contain a column with the SDSS objids and a column with the SDSS keyword "cModelFlux" (host galaxy flux in nanomaggies). 

The script roou.py then takes the original images from one of the three folders (fits_eval, fits_test, fits_train) and adds simulated AGN point sources to the galaxies in their centers. It also preprocesses the images, and saves them as .npy files so that they can be importet by the GAN.

The first step is modifying config.py. The parameters relevant for creating a training/testing/validation set are the following.
* ```redshift```: Unique identifier used to distinguish different datasets by redshift.
* ```filter_```: Unique identifier to distinguish different datasets by filters.
* ```ext```: Custom extension for folders to differentiate setups. This allows a User to create different test sets from the same original images.
* ```stretch_type``` and ```scale_factor```: Normalization function (and its parameter) applied to the images before saving them as .npy input for the GAN.
* ```pixel_min_value``` and  ```pixel_max_value```: Minimum and maximum pixel value accross the whole set of images. This is used for the normalization.
* ```uniform_logspace```: A boolean specifying wheter the contrast ratio should be distributed uniformly in linear or in logarithmic space.


You should modify some paths in roou.py:
* ```psfTool_path```: Directory containing the executable of the SDSS PSF tool path.
* ```psfFields_dir```: Directory containing the SDSS PSF metadata (psFields). This data is used as input for the SDSS PSF tool. 

Then you should modify a path in photometry.py
* ```fields_core_path```: Path to parent directory containing the SDSS fields (structured according to "run" and "camcol")

Finally you should modify a path in galfit.py
* ```galfit_path```: Path to your <tt>GALFIT</tt> binary file.


Then you can run roou.py. To create a training set, use:
```bash
python roou.py --mode 0    # Create a training set. Fit the SDSS tool PSF by three gaussians to model the PSF.

```

To create a test set, use:
```bash
python roou.py --mode 1 --mcombine 1    # Create a test set. Model the PSF by median-combining stars.
```

The following flags are available:
* ```mode```: If set to 0, the images from "fits_train" are used; if set to 1, the images from "fits_test" are used; if set to 2, the images from "fits_eval" are used.
* ```mcombine```: If set to False, the SDSS PSF tool is used to extract a PSF for each image. If mcombine is set to True, the PSF is extracted by median combining stars from the neighborhood of the galaxy.
* ```data_type```: If SDSS is used, ```data_type``` should be set to ```'sdss'```. If Hubble WFC3 F160W data is used, is should be set to ```'hubble'```.
* ```psf```: Whether the SDSS PSF or the Hubble PSF should be used to create fake AGN point sources.
* ```crop```: If set to 1, the GAN input (and output) images are cropped.
* ```save_psf```: If set to 1, for each galaxy the modeled PSF is saved as a .fits file.


### Train PSFGAN
```bash
python train.py
```
The path ```stretch_setup``` in config.py should point to the directory containing the folder ```npy_input```. The folder ```npy_input``` should contain the training input data (.npy files) in a subfolder called ```train```.

### Run a trained model
*We provide the pretrained models we describe in the paper. They were trained on 424 x 424 pixel SDSS r-band images in three different redshift ranges. To download the z~0.05 model, use the following command. To download the other models, just replace 0.05 by 0.1 or 0.2.*

```bash
wget https://share.phys.ethz.ch/~blackhole/spaceml/PSFGAN/pretrained_SDSS_z_0.05.tar.gz
```
*To run a pretrained model, perform the following steps.*

Modify the following constants in config.py.
* ```model_path```: Path to .ckpt file of pretrained model used in test.py. Should be an empty string if PSFGAN is used in training mode.
* ```use_gpu```: Value that "CUDA_VISIBLE_DEVICES" should be set to.

```bash
python test.py --mode test
```
Set the flag "mode" to "eval" if the validation data should be used instead of the test data.

The path ```stretch_setup``` in config.py should point to the directory containing the folder ```npy_input```. The folder ```npy_input``` should contain the testing (validation) input data (.npy files) in a subfolder called ```train``` (```eval```).
