import numpy as np
import astropy.io.fits as fits
import os
import glob
import logging
# Path to your GALFIT binary, needed to create a dataset based on SDSS data
galfit_path = ''
if galfit_path == '':
    logging.warn('In galfit.py : no path provided to your GALFIT binary.'\
                 'This might raise an error if trying to fit a PSF')
import photometry


def create_feedme_PSF(feedmefile, ifile, resultfile, ipath, platescale, centroid, imshape):
    xmin = 10
    ymin = 10
    xmax = imshape[1] - 10
    ymax = imshape[0] - 10
    file_res = open(ipath + '/' + str(feedmefile), "w")
    file_res.write('# IMAGE and GALFIT CONTROL PARAMETERS \n')
    file_res.write('A)  ' + str(ifile) + '  # Input data image (FITS file) \n')
    file_res.write('B)  ' + str(resultfile) + '  # Output data image block \n')
    file_res.write('C)  ' + 'none  # Sigma image name (made from data if blank or "none" \n')
    file_res.write('D)  ' + 'none  # Input PSF image and (optional) diffusion kernel \n')
    file_res.write('E)  ' + '1  # PSF fine sampling factor relative to data \n')
    file_res.write('F)  ' + 'none  # Bad pixel mask (FITS image or ASCII coord list \n')
    file_res.write('G)  ' + 'none  # File with parameter constraints (ASCII file)\n')
    file_res.write('H)  ' + str(xmin) + ' ' + str(xmax) + ' ' + str(ymin) + ' ' + str(
        ymax) + '  # Image region to fit (xmin xmax ymin ymax) \n')
    file_res.write('I)  ' + '1 1   # Size of the convolution box (x y)\n')
    file_res.write('J)  ' + str(22.3) + '   # Magnitude photometric zeropoint\n')
    file_res.write('K)  ' + str(platescale) + '  ' + str(platescale) + ' # Plate scale (dx dy)    [arcsec per pixel]\n')
    file_res.write('O)  ' + 'regular   # Display type (regular, curses, both) \n')
    file_res.write('P)  ' + '0   #Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps \n')
    file_res.write('# \n')
    file_res.write('#')

    file_res.write('# \n')
    file_res.write('# gaussian \n')
    file_res.write('0) ' + 'gaussian  # Component type \n')
    file_res.write('1) ' + str(centroid[0] + 1) + ' ' + str(centroid[1] + 1) + ' 1  1      # Position x, y \n')
    file_res.write('3) 14 1        #Integrated magnitude \n')
    file_res.write('4) 2   1       # FWHM [pix] \n')
    file_res.write('9) 1.0    1    # Axis ratio (b/a) \n')
    file_res.write('10) 25  1  # Position angle (PA) [deg: Up=0, Left=90] \n')
    file_res.write('Z) 0           # Skip this model in output image?  (yes=1, no=0) \n')
    file_res.write('# \n')

    file_res.write('# \n')
    file_res.write('# gaussian \n')
    file_res.write('0) ' + 'gaussian  # Component type \n')
    file_res.write('1) ' + str(centroid[0] + 1) + ' ' + str(centroid[1] + 1) + ' 1  1      # Position x, y \n')
    file_res.write('3) 14 1        #Integrated magnitude \n')
    file_res.write('4) 5   1       # FWHM [pix] \n')
    file_res.write('9) 1.0    1    # Axis ratio (b/a) \n')
    file_res.write('10) 25  1  # Position angle (PA) [deg: Up=0, Left=90] \n')
    file_res.write('Z) 0           # Skip this model in output image?  (yes=1, no=0) \n')
    file_res.write('# \n')

    file_res.write('# \n')
    file_res.write('# gaussian \n')
    file_res.write('0) ' + 'gaussian  # Component type \n')
    file_res.write('1) ' + str(centroid[0] + 1) + ' ' + str(centroid[1] + 1) + ' 1  1      # Position x, y \n')
    file_res.write('3) 14 1        #Integrated magnitude \n')
    file_res.write('4) 10   1       # FWHM [pix] \n')
    file_res.write('9) 1.0    1    # Axis ratio (b/a) \n')
    file_res.write('10) 25  1  # Position angle (PA) [deg: Up=0, Left=90] \n')
    file_res.write('Z) 0           # Skip this model in output image?  (yes=1, no=0) \n')
    file_res.write('# \n')

    file_res.write('# \n')
    file_res.write('# \n')
    file_res.write('0)  ' + 'sky \n')  # object type
    file_res.write('1)  ' + '0  0 \n')  # sky background at center of fitting region [ADUs]
    file_res.write('2)  ' + '0.000  0 \n')  # dsky/dx (sky gradient in x)
    file_res.write('3)  ' + '0.000  0 \n')  # dsky/dy (sky gradient in y)
    file_res.write('Z)  ' + '0 \n')  # output option (0 = resid., 1 = Don't subtract)
    file_res.close()



def open_GALFIT_results(file_path, framename):
    if framename == 'original':
        extension = 1
    elif framename == 'model':
        extension = 2
    elif framename == 'residual':
        extension = 3
    elif framename == 'all':
        hdu = fits.open(file_path)
        original = hdu[1].data
        model = hdu[2].data
        residual = hdu[3].data
        hdu.close()
        return original, model, residual
    else:
        raise ValueError('Please provide a framename which is either \
        	             "original","model", or "residual".')
    return fits.getdata(file_path, ext=extension)


def fit_PSF_GALFIT(fname, out_dir):
    """
    Args:
        fname: filename of PSF image
        out_dir: Directory where the fit should be saved to.
    Returns:
        PSF triple gaussian model.
    """
    print(fname)
    hdu = fits.open(fname)
    psf_data = np.array(hdu[0].data, dtype=float)
    imshape = psf_data.shape
    centroid = photometry.find_centroid(psf_data)
    print('PSF centroid = %s' % centroid)
    for f in glob.glob('%s/galfit.*' % out_dir):
        os.remove(f)
    # os.system('cd %s; rm -f galfit.*; rm -f result.fits'%fpath)
    out_name = os.path.basename(fname)
    out_path = '%s/%s' % (out_dir, out_name)
    create_feedme_PSF('galfit.feedme', fname, out_name, out_dir, 0.396,
    	              centroid, imshape)
    os.system('cd %s; %s galfit.feedme >> galfit.log' % (out_dir, galfit_path))
    if not os.path.exists(out_path):
        print('ERROR NO RESULT')
        return None
    return open_GALFIT_results(out_path, 'model')