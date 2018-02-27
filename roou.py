import argparse
import glob
import os
import random
import numpy as np
import pandas
from astropy.io import fits
from scipy.stats import norm
import photometry
from config import Config as conf
import normalizing
import astropy.table as astrotable
import logging

parser = argparse.ArgumentParser()

# Path to the folder containing 'read_PSF' SDSS PSF tool : http://www.sdss.org/dr12/algorithms/read_psf/
psfTool_path = ''
if psfTool_path == '':
    logging.warn('In roou.py : no path provided to your SDSS PSF tool.'\
                 'This might raise an error if trying to add a SDSS PSF')

# Path to PSF metadata (psFields) which the SDSS PSF tool uses as input.
psfFields_dir = conf.core_path+'/source/sdss/dr12/psf-data'

# Path to Hubble PSF if used:
# The following paths are only needed, if the Hubble PSF is used for simulating
# AGN point sources.
path_to_Hubble_psf = '/mnt/ds3lab/dostark/hubble_psf_many_stars.fits'
#path_to_Hubble_psf = '/mnt/ds3lab/dostark/PSFSTD_WFC3IR_F160W.fits'

# Directories for temporary files
tmpdir_for_SExtractor = conf.stretch_setup+'/tmp_for_SExtractor/'
tmp_GALFIT = conf.stretch_setup+'/tmp_for_GALFIT/'


def roou():
    random.seed(42)
    parser.add_argument("--output", default=conf.data_path)
    # mode: 0 for training, 1 for testing, 2 for validation
    parser.add_argument("--mode", default="1")
    parser.add_argument("--crop", default="0")
    parser.add_argument('--psf', default='sdss')
    parser.add_argument('--mcombine', default='0')
    parser.add_argument('--data_type', default='sdss')
    parser.add_argument('--save_psf', default="0")
    args = parser.parse_args()

    output = args.output
    mode = int(args.mode)
    cropsize = int(args.crop)
    psf_type = args.psf
    mcombine_int = int(args.mcombine)
    mcombine_boolean = bool(mcombine_int)
    data_type = args.data_type
    save_psf = bool(int(args.save_psf))

    # Conf parameters
    ratio_max = conf.max_contrast_ratio
    ratio_min = conf.min_contrast_ratio
    uniform_logspace = conf.uniform_logspace

    if mode == 1: #Test set
        input = '%s/fits_test' % conf.run_case
        catalog_path = glob.glob('%s/catalog_test*' % conf.run_case)[0]
        # Save the conditional input (galaxy + AGN) as .fits file before
        # stretching it.
        save_raw_input = True
    elif mode == 0: #Train set
        input = '%s/fits_train' % conf.run_case
        catalog_path = glob.glob('%s/catalog_train*' % conf.run_case)[0]
        if data_type == 'hubble':
            raise ValueError("No behaviour defined for mode=0, if data_type is \
                             'hubble'. ")
        save_raw_input = False
    elif mode == 2: #Validation set
        input = '%s/fits_eval' % conf.run_case
        catalog_path = glob.glob('%s/catalog_eval*' % conf.run_case)[0]
        if data_type == 'hubble':
            raise ValueError("No behaviour defined for mode=2, if data_type is \
                             'hubble'. ")
        save_raw_input = False
    print('Input files : %s' % input)

    # Read information (e.g. SDSS objid, path to image file, etc.) from catalog.
    if data_type == 'sdss':
        catalog = pandas.read_csv(catalog_path)
    elif data_type == 'hubble':
        table = astrotable.Table.read(catalog_path)
        df = table.to_pandas()
        catalog = df[['RAdeg', 'DECdeg', 'Hmag', 'IAU_Name']]

    train_folder = '%s/train' % output
    test_folder = '%s/test' % output
    eval_folder = '%s/eval' % output
    # raw_test_folde is the path to the folder where the unstretched input
    # images (galaxy + AGN) are saved as .fits files.
    raw_test_folder = '%s/fits_input%s' % (conf.run_case, conf.ext)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    if not os.path.exists(raw_test_folder):
        os.makedirs(raw_test_folder)
    filter_string = conf.filter_
    fits_path = '%s/*-%s.fits' % (input, filter_string)
    files = glob.glob(fits_path)

    # Iterate over all files in the directory 'input' and create the conditional
    # input (galaxy + PS) for them.
    not_found = 0
    pixel_max = 0
    for i in files:
        image_id = os.path.basename(i).replace('-'+filter_string+'.fits', '')
        print('\n')
        print(image_id)

        if data_type == 'sdss':
            obj_line = catalog.loc[catalog['dr7ObjID'] == int(image_id)]
        elif data_type == 'hubble':
            obj_line = catalog.loc[catalog['IAU_Name'] == str(image_id)]
        if obj_line.empty:
            not_found = not_found + 1
            print('Not found')
            continue

        f = i

        rfits = fits.open(f)
        data_r = rfits[0].data
        rfits.close()

        if data_type == 'sdss':
            flux = obj_line['cModelFlux_r'].item()
            if flux < 0:
                print "cModelFlux_r value in catalog is negative!"
                continue
        elif data_type == 'hubble':
            hmag = obj_line['Hmag'].item()
            # convert AB magnitudes to fluxes:
            # source (for zeropoint):
            # https://archive.stsci.edu/pub/hlsp/candels/goods-s/gs-tot/v1.0/
            # hlsp_candels_hst_acs-wfc3_gs-tot_readme_v1.0.pdf
            flux = 10**(-1./2.5*(hmag-25.9463))
        if data_type == 'sdss':
            fwhm = 1.4
            fwhm_use = fwhm / 0.396
        elif data_type == 'hubble':
            fwhm = 0.18
            fwhm_use = fwhm / 0.06

        # Sample the contrast ratios from the distribution specified in the file
        # config.py
        if uniform_logspace:
            r_exponent = random.uniform(np.log10(ratio_min),
                                        np.log10(ratio_max))
            r = 10**r_exponent
        else:
            r = random.uniform(ratio_min, ratio_max)
        print("ratio = %s" % r)


        if psf_type == 'sdss':
            if mcombine_boolean:
                data_PSF = photometry.add_sdss_PSF(i, data_r, r*flux, obj_line,
                                                   psfTool_path, psfFields_dir,
                                                   sexdir=tmpdir_for_SExtractor,
                                                   median_combine=True,
                                                   save_psf=save_psf)
            else:
                data_PSF = photometry.add_sdss_PSF(i, data_r, r*flux, obj_line,
                                                   psfTool_path, psfFields_dir,
                                                   save_psf=save_psf)
        elif psf_type == 'hubble':
            if mcombine_boolean:
                data_PSF = photometry.add_hubble_PSF(i, data_r, r*flux,
                                                     santinidir=catalog_path,
                                                     median_combine=True,
                                                     RA=obj_line['RAdeg'],
                                                     DEC=obj_line['DECdeg'],
                                                     GALFIT_tmpdir=tmp_GALFIT,
                                                     save_psf=save_psf)
            else:
                data_PSF = photometry.add_hubble_PSF(i, data_r, r*flux,
                                                     psfdir=path_to_Hubble_psf,
                                                     save_psf=save_psf)
        else:
            print('Unknown psf type : %s' % psf_type)
            raise ValueError(psf_type)

        if data_PSF is None:
            print('Ignoring file %s because PSF is missing.' % i)
            continue

        print('data_r centroid : %s' % photometry.find_centroid(data_r))
        print('data_PSF centroid : %s' % photometry.find_centroid(data_PSF))

        if (cropsize > 0):
            figure_original = np.ones((2*cropsize, 2*cropsize,
                                       conf.img_channel))
            figure_original[:, :, 0] = photometry.crop(data_r,
                                                       cropsize)
            figure_with_PSF = np.ones((2*cropsize, 2*cropsize,
                                       conf.img_channel))
            figure_with_PSF[:, :, 0] = photometry.crop(data_PSF,
                                                       cropsize)
        else:
            figure_original = np.ones((data_r.shape[0], data_r.shape[1],
                                       conf.img_channel))
            figure_original[:, :, 0] = data_r
            figure_with_PSF = np.ones((data_r.shape[0], data_r.shape[1],
                                       conf.img_channel))
            figure_with_PSF[:, :, 0] = data_PSF


        # Saving the "raw" data+PSF before stretching
        if save_raw_input:
            raw_name = '%s/%s-%s.fits' % (raw_test_folder, image_id,
                                          filter_string)
            # Overwrite if files already exist.
            hdu = fits.PrimaryHDU(data_PSF)
            hdu.writeto(raw_name, overwrite=True)

        # Preprocessing
        Normalizer = normalizing.Normalizer(stretch_type=conf.stretch_type,
                                            scale_factor=conf.scale_factor,
                                            min_value=conf.pixel_min_value,
                                            max_value=conf.pixel_max_value)
        figure_original = Normalizer.stretch(figure_original)
        figure_with_PSF = Normalizer.stretch(figure_with_PSF)

        # output result to pix2pix format
        figure_combined = np.zeros((figure_original.shape[0],
                                   figure_original.shape[1] * 2, 1))
        figure_combined[:, :figure_original.shape[1], :] = \
                                                        figure_original[:, :, :]
        figure_combined[:, figure_original.shape[1]:2*figure_original.shape[1],
                        :] = figure_with_PSF[:, :, :]

        if mode==1: # Testing set
            mat_path = '%s/test/%s-%s.npy' % (output, image_id, filter_string)
        elif mode==2: # Validation set
            mat_path = '%s/eval/%s-%s.npy' % (output, image_id, filter_string)
        elif mode==0: # Training set
            mat_path = '%s/train/%s-%s.npy' % (output, image_id, filter_string)
        np.save(mat_path, figure_combined)

        if np.max(photometry.crop(data_PSF, 20)) > pixel_max:
                  pixel_max = np.max(photometry.crop(data_PSF, 20))

    print('Maximum pixel value inside a box of 40x40 pixels around the center:')
    print(pixel_max)
    print("%s images have not been used because there were no corresponding" \
           " objects in the catalog") % not_found


if __name__ == '__main__':
    roou()
