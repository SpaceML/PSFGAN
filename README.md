# PSFGAN

This project is the implementation of the paper "<tt>PSFGAN</tt>: a generative adversarial network system for separating quasar point sources and host galaxy light". This code can be used to create images for training sets and to apply pretrained models to new data.

## Prerequisites
Linux or OSX

NVIDIA GPU + CUDA CuDNN for training on GPU's

We further use the following tools to create training and testing sets:
<tt>GALFIT</tt>, <tt>Source Extractor</tt>, SDSS PSF tool ("Stand Alone Code" downloaded from http://www.sdss.org/dr12/algorithms/read_psf/)

## Dependencies
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
The first step always is modifying the file config.py. The most important parameters are the following:
* core_path: Path to directory containing all the subdirectories.
* redshift: Unique redshift identifier used for naming the subdirectories.
* filter_: Unique identifier to distinguish images observed in different filters.
* model_path: Path to .ckpt file of pretrained model used in test.py. Should be an empty string if PSFGAN is used in training mode.
* use_gpu: Value that "CUDA_VISIBLE_DEVICES" should be set to.

### Create training/validation/testing sets
If you already have training/testing data or you are applying PSFGAN to real data, you can skip everything described in this section.

The code assumes a directory structure like the following.
```bash
core_path/
├── z_0.1
│   └── r-band
│       ├── catalog_z_0.095_0.105.csv
│       ├── fit_eval
│       ├── fits_test
│       └── fits_train
└── z_0.05
    └── r-band
        ├── catalog_z_0.045_0.055.csv
        ├── fits_eval
        ├── fits_test
        └── fits_train
```
In this example there are two subfolders containing training, testing and validation images and a catalog. For SDSS data the catalog must contain a column with the SDSS objids and a column with the SDSS keyword "cModelFlux" (host galaxy flux in nanomaggies). 

The script roou.py then takes the original .fits images from one of the three folders (fits_eval, fits_test, fits_train) and adds simulated AGN point sources to the galaxies in their centers. It also preprocesses the images, and saves them as .npy files so that they can be importet by the GAN.

You should modify some paths in roou.py:
* tmpdir_for_SExtractor: Directory where temporary files for SExtractor are saved. This is only needed if the flag "mcombine" is set to 1.
* psfTool_path: Directory containing the executable of the SDSS PSF tool path.
* psfFields_dir: Directory containing the SDSS PSF metadata (psFields). This data is used as input for the SDSS PSF tool. 

Then you should modify some paths in photometry.py
* galfit_command: Path to <tt>GALFIT</tt> executables.
* fields_core_path: Path to parent directory containing the SDSS fields (structured according to "run" and "camcol")

Then you can run roou.py:

```bash
python roou.py --mode 1 --mcombine 1     # Create a test and median combine stars to extract the PSF.
```

* mode: If set to 0, the images from "fits_train" are used; if set to 1, the images from "fits_test" are used; if set to 2, the images from "fits_eval" are used.
* mcombine: If set to False, the SDSS PSF tool is used to extract a PSF for each image. If mcombine is set to True, the PSF is extracted by median combining stars from the neighborhood of the galaxy.

### Train PSFGAN
```bash
python train.py
```

### Run a trained model
Before you run test.py you should modifiy "model_path" in config.py. Set the flag "mode" to "eval" if the validation data should be used instead of the test data.

```bash
python test.py --mode test
```
