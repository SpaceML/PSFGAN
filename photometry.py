import numpy as np
from astropy.io import fits
import os
import galfit
from galfit import galfit_path as galfit_command
from config import Config as conf
from photutils import centroid_com
import photutils
from astropy.stats import sigma_clipped_stats
import subprocess
import glob
import astropy.table as astrotable
from astropy import wcs
from photutils import DAOStarFinder
import random

sex = 'sextractor'
fields_core_path = conf.core_path+'/source/sdss/dr12/plates/'

def calc_zeropoint(exposure_time, calibration_factor):
    return 22.5 + 2.5 * np.log10(1. / exposure_time / calibration_factor)


def mag_from_counts(counts, t_exp, zeropoint):
    if counts > 0:
        return -2.5*np.log10(counts/t_exp) + zeropoint
    else:
        print '\n'
        print 'negative count value!!!'
        print '\n'
        return -999


def crop(img, cut):
    """
    Crop the image to a cut*cut image centered on the center
    """
    if img.shape[0]!=img.shape[1]:
        quit()
    size = img.shape[0]
    return img[size / 2 - cut:size / 2 + cut, size / 2 - cut:size / 2 + cut]


def find_centroid(img, cut=5, center=None):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    x_tmp, y_tmp = centroid_com(crop(img, cut))
    return [center[0] - cut + x_tmp, center[1] - cut + y_tmp]


def starphot(imdata, position, radius, r_in, r_out):
    """
    sources: http://photutils.readthedocs.io/en/stable/
    photutils/aperture.html
    ARGS:
        imdata:   Numpy array containing the star.
        position: [x,y] coordinates where x corresponds to the second index of
                  imdata
        radius:   Radius of the circular aperture used to compute the flux of
                  the star.
        r_in:     Inner radius of the annulus used to compute the background
                  mean.
        r_out:    Outer radius of the annulus used to compute the background
                  mean.
    Returns [flux, background variance, background mean]
    """
    try:
        statmask = photutils.make_source_mask(imdata, snr=5, npixels=5,
                                              dilate_size=10)
    except TypeError:
        return None
    bkg_annulus = photutils.CircularAnnulus(position, r_in, r_out)
    bkg_phot_table = photutils.aperture_photometry(imdata, bkg_annulus,
                                                   method='subpixel',
                                                   mask=statmask)
    bkg_mean_per_pixel = bkg_phot_table['aperture_sum'] / bkg_annulus.area()
    src_aperture = photutils.CircularAperture(position, radius)
    src_phot_table = photutils.aperture_photometry(imdata, src_aperture,
                                                   method='subpixel')
    signal = src_phot_table['aperture_sum'] - bkg_mean_per_pixel*\
                                              src_aperture.area()
    #noise_squared = signal + bkg_mean_per_pixel*src_aperture.area()
    mean, median, std = sigma_clipped_stats(imdata, sigma=3.0, iters=5,
                                            mask=statmask)
    noise_squared = std**2
    return float(str(signal.data[0])), noise_squared,\
           float(str(bkg_mean_per_pixel.data[0]))

def weighted_median(data, weights):
    """
    This function omputes the weighted median of a sequence according to
    https://en.wikipedia.org/wiki/Weighted_median.
    Args:
        data: 1d numpy array of data
        weights: 1d numpy array of weights
    """
    weights_normalized = weights/weights.sum()
    sorted_data = np.sort(data)
    weights_parallel_sorted = weights_normalized[data.argsort()]
    sum = 0
    for i in range(0, len(weights_parallel_sorted)):
        actual_weight = weights_parallel_sorted[i]
        if sum + actual_weight >= 0.5:
            if sum + actual_weight > 0.5:
                return sorted_data[i]
            elif sum + actual_weight == 0.5:
                return 0.5*(sorted_data[i+1]+sorted_data[i])
        sum += actual_weight


def weighted_median_stacking(images, weights):
    imshape = images[0].shape
    combined_image = np.zeros(imshape)
    for x in range(0, imshape[1]):
        for y in range(0, imshape[0]):
            im_tmp = []
            for k in range(0, len(images)):
                im_tmp.append(images[k][y, x])
            combined_image[y, x] = weighted_median(np.array(im_tmp),
                                                   np.array(weights))
    return combined_image


def get_stars_from_field(tmp_path, field_filename, SExtractor_params,
                         imageshape, edge, mindist):
    """
    Args:
        tmp_path:          Path to directory where temporary SExtractor files
                           should be saved. Those files are deleted after
                           SExtractor has done its job.
        field_filename:    Filename of the SDSS field (.fits image). This image
                           should be in tmp_path.
        SExtractor_params: A dict of the used SExtractor input parameters,
                           denoted by the following keywords: magzero,
                           threshold, saturation_level, gain, pixel_scale, fwhm.
        imageshape:        Shape (y, x) of the field image where the first
                           index corresponds to the length in y direction and
                           the second index to the length in x direction.
        edge:              Margin where no stars are selected.
        mindist:           Minimal distance to other detections a star must
                           have in order to be selected.
    Returns:
        Returns the coordinates of the selected stars as an array of
        x-coordinates and an array of y-coordinates. Also returns the fluxes of
        the stars in an array of the same order. Finally returns a bool that
        indicates whether stars have been found at all.
    """
    file_res = open(tmp_path + 'sex_stars.conf', "w")
    file_res.write('#-------------------------------- Catalog ------------------------------------\n\n')
    file_res.write('CATALOG_NAME     sex_stars.cat        # name of the output catalog\n')
    file_res.write('CATALOG_TYPE     ASCII_HEAD     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n')
    file_res.write('                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC \n')
    file_res.write('PARAMETERS_NAME  {}{}           # name of the file containing catalog contents \n\n'.format(tmp_path, 'sex_stars.param'))
    file_res.write('#------------------------------- Extraction ----------------------------------\n\n')
    file_res.write('DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)\n')
    file_res.write('DETECT_MINAREA   5              # min. # of pixels above threshold\n')
    file_res.write('DETECT_THRESH    5              # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n')
    file_res.write('ANALYSIS_THRESH  {}             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n\n'.format(SExtractor_params['threshold']))
    file_res.write('FILTER           Y              # apply filter for detection (Y or N)?\n')
    file_res.write('FILTER_NAME      /mnt/ds3lab/dostark/sextractor_defaultfiles/default.conv   # name of the file containing the filter\n\n')
    file_res.write('DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds \n')
    file_res.write('DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending\n\n')
    file_res.write('CLEAN            Y              # Clean spurious detections? (Y or N)?\n')
    file_res.write('CLEAN_PARAM      1.0            # Cleaning efficiency)\n\n')
    file_res.write('MASK_TYPE        CORRECT        # type of detection MASKing: can be one of\n\n')
    file_res.write('                                # NONE, BLANK or CORRECT\n\n')
    file_res.write('#------------------------------ Photometry -----------------------------------\n\n')
    file_res.write('PHOT_APERTURES   5              # MAG_APER aperture diameter(s) in pixels\n')
    file_res.write('PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>\n')
    file_res.write('PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,\n')
    file_res.write('                                # <min_radius>\n\n')
    file_res.write('SATUR_LEVEL      {}             # level (in ADUs) at which arises saturation\n'.format(SExtractor_params['saturation_level']))
    file_res.write('SATUR_KEY        SATURATE       # keyword for saturation level (in ADUs)\n\n')
    file_res.write('MAG_ZEROPOINT    {}             # magnitude zero-point\n'.format(SExtractor_params['magzero']))
    file_res.write('MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)\n')
    file_res.write('GAIN             {}             # detector gain in e-/ADU\n'.format(SExtractor_params['gain']))
    file_res.write('GAIN_KEY         GAIN           # keyword for detector gain in e-/ADU\n')
    file_res.write('PIXEL_SCALE      {}             # size of pixel in arcsec (0=use FITS WCS info)\n\n'.format(SExtractor_params['pixel_scale']))
    file_res.write('# ------------------------- Star/Galaxy Separation ----------------------------\n\n')
    file_res.write('SEEING_FWHM      {}             # stellar FWHM in arcsec\n'.format(SExtractor_params['fwhm']))
    file_res.write('STARNNW_NAME     /mnt/ds3lab/dostark/sextractor_defaultfiles/default.nnw  # Neural-Network_Weight table filename\n\n')
    file_res.write('# ------------------------------ Background -----------------------------------\n\n')
    file_res.write('BACK_SIZE        64              # Background mesh: <size> or <width>,<height>\n')
    file_res.write('BACK_FILTERSIZE  3               # Background filter: <size> or <width>,<height>\n\n')
    file_res.write('BACKPHOTO_TYPE   GLOBAL          # can be GLOBAL or LOCAL\n\n')
    file_res.write('#------------------------------ Check Image ----------------------------------\n\n')
    file_res.write('CHECKIMAGE_TYPE  NONE            # can be NONE, BACKGROUND, BACKGROUND_RMS,\n')
    file_res.write('                                 # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,\n')
    file_res.write('                                 # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,\n')
    file_res.write('                                 # or APERTURES\n')
    file_res.write('CHECKIMAGE_NAME  /mnt/ds3lab/dostark/sextractor_defaultfiles/check.fits     # Filename for the check-image\n\n')
    file_res.write('#--------------------- Memory (change with caution!) -------------------------\n\n')
    file_res.write('MEMORY_OBJSTACK  3000            # number of objects in stack\n')
    file_res.write('MEMORY_PIXSTACK  300000          # number of pixels in stack\n')
    file_res.write('MEMORY_BUFSIZE   1024            # number of lines in buffer\n\n')
    file_res.write('#----------------------------- Miscellaneous ---------------------------------\n\n')
    file_res.write('VERBOSE_TYPE     NORMAL          # can be QUIET, NORMAL or FULL\n')
    file_res.write('HEADER_SUFFIX    .head           # Filename extension for additional headers\n')
    file_res.write('WRITE_XML        N               # Write XML file (Y/N)?\n')
    file_res.write('XML_NAME         sex.xml         # Filename for XML output\n')
    file_res.close()

    file_param = open(tmp_path+'sex_stars.param', "w")
    file_param.write('NUMBER\n')
    file_param.write('X_IMAGE\n')
    file_param.write('Y_IMAGE\n')
    file_param.write('FLUX_AUTO\n')
    file_param.write('CLASS_STAR\n')
    file_param.write('FWHM_IMAGE\n')
    file_param.write('FLAGS\n')
    file_param.close()

    # run SExtractor & read results
    os.system('cd '+tmp_path+ ' ; '+sex+' -c sex_stars.conf '+field_filename)
    try:
        data = np.genfromtxt(tmp_path+'sex_stars.cat',dtype=None,comments='#',
                             names=['number', 'x', 'y', 'flux', 'classifier',
                                    'fwhm', 'flags'])
    except IOError:
        return [], [], [], False
    x_data = np.array(data['x'])
    y_data = np.array(data['y'])
    fluxes = np.array(data['flux'])
    star_class_data = np.array(data['classifier'])
    fwhm_data = np.array(data['fwhm'])
    flags = np.array(data['flags'], dtype=int)

    # discard useless detection
    mask = (x_data >= edge) & (y_data >= edge) & \
                              (x_data <= imageshape[1] - edge) & \
                              (y_data <= imageshape[0] - edge) & (flags!=4)
    mask_negative = np.invert(mask)
    star_class_data[mask_negative] *= 0.0
    fwhm_deviations_from_min = fwhm_data - np.min(fwhm_data)
    for i in range(0, len(x_data)):
        for j in range(i+1, len(x_data)):
            if np.sqrt((x_data[i]-x_data[j])**2 +
                       (y_data[i]-y_data[j])**2) < mindist:
                star_class_data[i] = 0
                star_class_data[j] = 0
        if fwhm_deviations_from_min[i] > 4:
            star_class_data[i] = 0
    starmask = star_class_data >= 0.9
    stars_xcoords = x_data[starmask]
    stars_ycoords = y_data[starmask]
    fluxes = fluxes[starmask]
    return np.array(stars_xcoords), np.array(stars_ycoords), np.array(fluxes), \
           np.max(star_class_data)>=0.9

def run_sdss_psftool(obj_line, psf_fname, SDSStool_path, psFields_path):
    """
    Args:
        obj_line:       Dictionary containing the following SDSS parameters
                        keywords: run, rerun, camcol, field, colc, rowc.
        psf_fname:      Full path (including desired filename) of the output
                        image.
        SDSStool_path:  Path to directory containing the stand alone code of the
                        SDSS PSF tool (http://www.sdss.org/dr12/algorithms/
                        read_psf/).

        psFields_path: Path to the (downloaded) PSF metadata provided by SDSS.
    Returns:
        No value returned. The PSF image is created at 'psf_fname'.
    """
    # Get the input parameters for the tool.
    psfTool_path = '%s/read_PSF' % SDSStool_path
    psfFields_dir_1 = psFields_path

    filter_string = conf.filter_
    run = obj_line['run'].item()
    rerun = obj_line['rerun'].item()
    camcol = obj_line['camcol'].item()
    field = obj_line['field'].item()

    psfField = '%s/psField-%06d-%d-%04d.fit' % (psfFields_dir_1, run, camcol,
                                                field)
    if not os.path.exists(psfField):
        psfField = '%s/%d/%d/objcs/%d/psField-%06d-%d-%04d.fit' % (
            psfFields_dir_1, rerun, run, camcol, run, camcol, field)
    if not os.path.exists(psfField):
        raise OSError('No psfField fit found')

    colc = obj_line['colc'].item()
    rowc = obj_line['rowc'].item()
    filter_dic = {'u': 1, 'g':2, 'r':3, 'i':4, 'z':5}

    # Run the code and read in the output image.
    os.system('%s %s %s %s %s %s' % (psfTool_path, psfField,
                                     filter_dic[filter_string], rowc, colc,
                                     psf_fname))
    try:
        hdu = fits.open(psf_fname)
        psf_data = np.array(hdu[0].data, dtype=float) / 1000 - 1
        hdu.close()
        os.remove(psf_fname)
        hdu = fits.PrimaryHDU(psf_data)
        hdu.writeto(psf_fname)
    except:
        print('no psf %s' % psf_filename)
        print('psfField = %s' % psfField)
        print('rowc = %s' % rowc)
        print('colc = %s' % colc)

def get_field(obj_line):
    """
    ATTENTION: This function requires the directories containing the SDSS fields
    to be structured in a specific way!.
    Args_
        obj_line:  Dictionary containing the following SDSS parameters keywords:
                   run, rerun, camcol, field, colc, rowc.
    Returns:
        Returns the desired SDSS field as a numpy array.
    """
    core_path = fields_core_path
    filter_string = conf.filter_
    run = obj_line['run'].item()
    rerun = obj_line['rerun'].item()
    camcol = obj_line['camcol'].item()
    field = obj_line['field'].item()
    relative_path = '%s/%s' %(run, camcol)

    if not os.path.isfile('%s%s/sdss%s_dr12_%s-%s.fits.bz2' % (core_path,
                                                               relative_path,
                                                               filter_string,
                                                               run, field)):
        raise OSError('File sdss%s_dr12_%s-%s.fits.bz2 not found in %s'%(relative_path,
                                                                         filter_string,
                                                                         run, field,
                                                                         core_path))
    os.system('cd %s%s; bzip2 -dk sdss%s_dr12_%s-%s.fits.bz2' % (core_path,
                                                                 relative_path,
                                                                 filter_string,
                                                                 run, field))
    try:
        data = fits.getdata('%s%s/sdss%s_dr12_%s-%s.fits' % (core_path,
                                                             relative_path,
                                                             filter_string,
                                                             run, field))
        os.system('rm %s%s/sdss%s_dr12_%s-%s.fits' % (core_path, relative_path,
                                                      filter_string, run,
                                                      field))
    except IOError:
        print 'file does not exist'
        return None
    return data


def GALFIT_fit_stars(tmpdir, data, phot_zeropoint, platescale, mag_guess,
                     fwhm_guess):
    """
    This function fits a cutout star with one gaussian+sky to find its center.

    Args:
        tmpdir:          Path to directory where temporary files should be saved
                         (and deleted).
        data:            Cutout of a star as a numpy array.
        phot_zeropoint:  Photometric zeropoint.
        platescale:      Pixel scale to convert between arcseconds and image
                         coordinates.
        mag_guess:       Initial guess of the magnitude of the star.
        fwhm_guess:      Initial guess of the FWHM of the gaussian PSF.
    Returns:
        X: X-coordinate of the mean of the gaussian.
        Y: Y-coordinate of the mean of the gaussian.
    """
    hdu_output = fits.PrimaryHDU(data)
    hdulist_output = fits.HDUList([hdu_output])
    hdulist_output.writeto(tmpdir+'star.fits', overwrite=True)

    file_res = open(tmpdir + 'galfit.feedme', "w")
    file_res.write('# IMAGE and GALFIT CONTROL PARAMETERS \n')
    file_res.write('A)  ' + 'star.fits            # Input data image (FITS file) \n')
    file_res.write('B)  ' + 'result.fits' + '           # Output data image block\n')
    file_res.write('C)  ' + 'none                    # Sigma image name (made from data if blank or "none") \n')
    file_res.write('D)  ' + 'none                # Input PSF image and (optional) diffusion kernel\n')
    file_res.write('E)  1                     # PSF fine sampling factor relative to data\n')
    file_res.write('F)  ' + 'none                # Bad pixel mask (FITS image or ASCII coord list)\n')
    file_res.write('G)  ' + 'none                      # File with parameter constraints (ASCII file)\n')
    file_res.write('H)  ' +'1  '+str(data.shape[1]) + '  1  '+str(data.shape[0]) + '        # Image region to fit (xmin xmax ymin ymax) \n')
    file_res.write('I)  ' + str(data.shape[1]) + '  ' + str(data.shape[0]) + '                  # Size of the convolution box (x y)\n')
    file_res.write('J)  ' + str(phot_zeropoint) + '                        # Magnitude photometric zeropoint \n')
    file_res.write('K)  ' + str(platescale) + '  ' + str(platescale) + '                       # Plate scale (dx dy)    [arcsec per pixel]\n')
    file_res.write('O)  ' + 'regular                        # Display type (regular, curses, both)\n')
    file_res.write('P)  ' + '0                               #Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps \n')
    file_res.write('# \n\n')
    file_res.write('# \n\n')

    file_res.write('# Component number 1\n')
    file_res.write('0)  ' + 'gaussian \n')
    file_res.write('1)  ' + str(data.shape[1]/2) + '  ' + str(data.shape[0]/2) + '  1  1           # position x, y        [pixel] \n')
    file_res.write('3)  ' + str(mag_guess) + '  1                 # total magnitude of point source \n')
    file_res.write('4)  ' + str(fwhm_guess) + '  1                 # fwhm \n')
    file_res.write('9)  ' + '1  0                 # axis ratio \n')
    file_res.write('Z)  ' + '0 \n')
    file_res.write('# \n')
    file_res.write('# \n')

    file_res.write('# Sky Component\n')
    file_res.write('0)  ' + 'sky                #  object type \n')
    file_res.write('1)  ' + '0  1            #  sky background at center of fitting region [ADUs] \n')
    file_res.write('2)  ' + '0.000  0                #  dsky/dx (sky gradient in x)\n')
    file_res.write('3)  ' + '0.000  0              #  dsky/dy (sky gradient in y) \n')
    file_res.write('Z) ' + '1 \n')  # output option (0 = resid., 1 = Don't subtract)
    file_res.close()

    output = subprocess.check_output(["cd "+str(tmpdir)+" ; "+galfit_command+
                                      " galfit.feedme"], shell=True,
                                      stderr=subprocess.STDOUT)
    try:
        hdu= fits.open(tmpdir+'result.fits')
    except IOError:
        return None
    model_header = hdu[2].header
    hdu.close()
    X = float(model_header['1_XC'][:6])
    Y = float(model_header['1_YC'][:6])
    os.remove(tmpdir+'result.fits')
    os.remove(tmpdir+'star.fits')
    os.remove(tmpdir+'fit.log')
    os.remove(tmpdir+'galfit.01')
    return X, Y


def empirical_PSF(data, x_icords, y_icords, cutout_size, fwhm_pix, galfit_tmp,
                  mag_guesses, phot_zeropoint, pixel_scale):
    """
    This function extracts an empirical PSF by median combining stars whose
    coordinates are given as an input.

    Args:
        data:           Image (as numpy array) containing the stars.
        x_icords:       X-coordinates (second index) of stars that are being
                        combined.
        y_icords:       Y-coordinates (second index) of stars that are being
                        combined.
        cutout_size:    Size of the resulting PSF image.
        fwhm_pix:       Estimate of the FWHM of the PSF in pixels.
        galfit_tmp:     Path to directory containing temporary files GALFIT uses
                        or creates to find the centroids of the stars. This
                        directory is emptied after GALFIT has done its job.
        mag_guesses:    Magnitude guesses for the stars. They are used as input
                        for GALFIT. GALFIT is used to accurately fit the center
                        of the source.
        phot_zeropoint: Zeropoint of the used magnitude system.
        pixel_scale:    Arcseconds/pixel
    """
    # Rough cutout of stars (with background!) and weights (S/N^2) calculation.
    cutouts = []
    weights = []
    xc = []
    yc = []
    integrated_counts = []
    cutout_size_tmp = cutout_size + 5
    for j in range(0, len(x_icords)):
        cutout_candidate = data[int(y_icords[j]) - cutout_size_tmp:
                                int(y_icords[j]) + cutout_size_tmp+1,
                                int(x_icords[j]) - cutout_size_tmp:
                                int(x_icords[j]) + cutout_size_tmp+1]
        # Test if cutout is empty (sometimes occurs for the goods-s data):
        if np.mean(cutout_candidate)==0.0:
            continue
        # Double check wheter there is a star in the center of the cutout.
        mean, median, std = sigma_clipped_stats(cutout_candidate, sigma=3.0,
                                                iters=5)
        daofind = DAOStarFinder(fwhm=fwhm_pix, threshold=5.*std)
        sources = daofind(cutout_candidate - median)
        detections_x = np.array(sources['xcentroid'])
        detections_y = np.array(sources ['ycentroid'])
        distances = np.sqrt((detections_x-cutout_size_tmp)**2+
                            (detections_y-cutout_size_tmp)**2)
        if (not list(distances)) or (np.min(distances) > 2):
            continue
        # if candidate has passed the tests, it is added to the stack.
        cutouts.append(cutout_candidate)
        fit_params = GALFIT_fit_stars(galfit_tmp, cutouts[-1], phot_zeropoint,
                                      pixel_scale, mag_guesses[j], fwhm_pix)

        xc.append(fit_params[0])
        yc.append(fit_params[1])
        starphot_results = starphot(cutouts[-1], fit_params, fwhm_pix*3,
                                    r_in=fwhm_pix*3, r_out=cutout_size_tmp-1)
        if starphot_results:
            S, N2 = starphot_results[0], starphot_results[1]
            integrated_counts.append(S)
        else:
            return None
        weights.append(S/N2)

    #recentering
    i = 0
    weights_final = []
    cutouts_recentered = []
    for image in cutouts:
        star_centroid = [xc[i], yc[i]]
        tmp = image
        if star_centroid[0] > cutout_size and tmp.shape[1] - cutout_size > \
                              star_centroid[0] and star_centroid[1] > \
                              cutout_size and tmp.shape[0] - cutout_size > \
                              star_centroid[1]:
            cutouts_recentered.append(tmp[int(star_centroid[1]) - cutout_size:
                                          int(star_centroid[1]) + cutout_size+1,
                                          int(star_centroid[0]) - cutout_size:
                                          int(star_centroid[0]) + cutout_size+1]
                                          /integrated_counts[i])
            weights_final.append(weights[i])
        i += 1
    # combine using the weighted median with weights of S/N^2.
    if len(weights_final)==0:
        return None
    print str(len(weights_final)) +' stars combined.'
    return weighted_median_stacking(cutouts_recentered, weights_final)


def add_sdss_PSF(origpath, original, psf_flux, obj_line, SDSStool_path,
                 psFields_path, sexdir=None, median_combine=False,
                 save_psf=False):
    """
    Args:
        origpath:      Path to original images.
        psf_flux:      Desired flux of the PSF that is beeing added to the
                       original image.
        obj_line:      Dictionary containing the following SDSS parameters
                       keywords: run, rerun, camcol, field, colc, rowc, dr7ObjID.
        SDSStool_path:  Path to SDSS tool (stand alone code) executable.
        psFields_path: Path to PSF meda data (psFields).
        sexdir:        Path where temporary SExtractor files should be saved.
                       This directory is emptied after SExtractor has done its
                       job.
    """
    SDSS_psf_dir = '%s/psf/SDSS' % conf.run_case
    GALFIT_psf_dir = '%s/psf/GALFIT' % conf.run_case
    filter_string = conf.filter_
    if not os.path.exists(SDSS_psf_dir):
        os.makedirs(SDSS_psf_dir)
    if not os.path.exists(GALFIT_psf_dir):
        os.makedirs(GALFIT_psf_dir)

    obj_id = obj_line['dr7ObjID'].item()
    SDSS_psf_filename = '%s/%s-%s.fits' % (SDSS_psf_dir, obj_id, filter_string)
    GALFIT_psf_filename = '%s/%s-%s.fits' % (GALFIT_psf_dir, obj_id,
                                             filter_string)
    if not os.path.exists(GALFIT_psf_filename):
        if not os.path.exists(SDSS_psf_filename):
            run_sdss_psftool(obj_line, SDSS_psf_filename, SDSStool_path,
                             psFields_path)
        # Fit the SDSS tool PSF with 3 gaussians to get rid of the noise.
        psf = galfit.fit_PSF_GALFIT(SDSS_psf_filename, GALFIT_psf_dir)
        if psf is None:
            print('Error in Galfit fit')
            return None
    else:
        psf = galfit.open_GALFIT_results(GALFIT_psf_filename, 'model')

    # median combine stars to get the PSF image if median_combine==True
    if median_combine:
        if not os.path.isdir(sexdir):
            os.makedirs(sexdir)
        # Try to get the conversion from nanomaggies to counts. It the
        # information is not available, just use one single value.
        try:
            nmgy_per_count = fits.getheader(origpath)['NMGY']
        except KeyError:
            nmgy_per_count = 0.0238446
        field_data = get_field(obj_line)
        # Get the SDSS field. If the data is not available, return None such
        # that this objid is skipped in roou.py
        if field_data is None:
            return None
        hdu_output = fits.PrimaryHDU(field_data/nmgy_per_count)
        hdulist_output = fits.HDUList([hdu_output])
        hdulist_output.writeto(sexdir+'field_ADU.fits', overwrite=True)
        # Some SDSS parameters for SExtractor.
        exptime = 53.9
        threshold = 5
        saturation_limit = 4000
        gain = 4.73
        pixel_scale = 0.396
        fwhm = 1.4
        zeropoint = calc_zeropoint(exptime, nmgy_per_count)
        sex_edge = 26
        SExtractor_params={'exptime': exptime, 'threshold': threshold,
                           'saturation_level': saturation_limit, 'gain': gain,
                           'pixel_scale': pixel_scale, 'fwhm': fwhm,
                           'magzero': zeropoint}
        x_coordinates, y_coordinates, fluxes, starboolean = \
                                      get_stars_from_field(sexdir,
                                                           'field_ADU.fits',
                                                           SExtractor_params,
                                                           field_data.shape,
                                                           sex_edge,
                                                           mindist=40)
        if not starboolean:
            files_to_delete = glob.glob(sexdir+'*')
            for f in files_to_delete:
                os.remove(f)
            return None


        csize = 20
        fluxes = fluxes*nmgy_per_count
        fluxmask = fluxes > 0
        mag_guesses = []
        for f in fluxes:
            mag_guesses.append(mag_from_counts(f/nmgy_per_count,
                                               exptime, zeropoint))
        mag_guesses = np.array(mag_guesses)
        # get median combined PSF
        psf_unscaled = empirical_PSF(field_data, x_coordinates[fluxmask],
                                     y_coordinates[fluxmask], csize,
                                     fwhm/pixel_scale, sexdir,
                                     mag_guesses[fluxmask], zeropoint,
                                     pixel_scale)
        if psf_unscaled is None:
            files_to_delete = glob.glob(sexdir+'*')
            for f in files_to_delete:
                os.remove(f)
            return None

        # compute statistics to subtract the background
        psf_centroid = [csize, csize]
        try:
            statmask = photutils.make_source_mask(psf_unscaled, snr=5,
                                                  npixels=5, dilate_size=10)
        except TypeError:
            files_to_delete = glob.glob(sexdir+'*')
            for f in files_to_delete:
                os.remove(f)
            return None
        bkg_annulus = photutils.CircularAnnulus(psf_centroid, 3*fwhm /
                                                pixel_scale, 20)
        bkg_phot_table = photutils.aperture_photometry(psf_unscaled,
                                                       bkg_annulus,
                                                       method='subpixel',
                                                       mask = statmask)
        bkg_mean_per_pixel = bkg_phot_table['aperture_sum'] / bkg_annulus.area()
        src_aperture = photutils.CircularAperture(psf_centroid, 3*fwhm / \
                                                  pixel_scale)
        src_phot_table = photutils.aperture_photometry(psf_unscaled,
                                                       src_aperture,
                                                       method='subpixel')
        flux_photutils = src_phot_table['aperture_sum'] - bkg_mean_per_pixel * \
                                                          src_aperture.area()
        scale_factor = psf_flux / flux_photutils
        psf = scale_factor * (psf_unscaled-bkg_mean_per_pixel)

        files_to_delete = glob.glob(sexdir+'*')
        for f in files_to_delete:
            os.remove(f)

    # else <==> median_combine==False
    else:
        # Use the 3 gaussian fit of the PSF generated by the SDSS PSF tool.
        psf = psf / psf.sum()
        psf = psf * psf_flux

    center = [original.shape[1] // 2, original.shape[0] // 2]
    centroid_galaxy = find_centroid(original)
    centroid_PSF = find_centroid(psf)

    composite_image = np.copy(original)

    gal_x = int(centroid_galaxy[0])
    gal_y = int(centroid_galaxy[1])
    ps_x = int(centroid_PSF[0])
    ps_y = int(centroid_PSF[1])

    # Put the PS on top of the galaxy at its centroid.
    for x in range(0, psf.shape[1]):
        for y in range(0, psf.shape[0]):
            x_rel = gal_x - ps_x + x
            y_rel = gal_y - ps_y + y
            if x_rel >= 0 and y_rel >= 0 and x_rel < original.shape[1] and \
                                                      y_rel < original.shape[0]:
                composite_image[y_rel, x_rel] += psf[y, x]

    return composite_image


def add_hubble_PSF(origpath, original, psf_flux, santinidir=None,
                   median_combine=False, psfdir=None, RA=None, DEC=None,
                   GALFIT_tmpdir=None, save_psf=False):
    """
    Args:
        origpath:      Path to original images.
        psf_flux:      Desired flux of the PSF that is beeing added to the
                       original image.
        obj_line:      Dictionary containing the following SDSS parameters
                       keywords: run, rerun, camcol, field, colc, rowc, dr7ObjID.
        tabledir:      Path to Santini table. This is needed to read the
                       positions of stars if median_combine is True.
        median_combine:Bool specifying wheter a single Hubble PSF is used
                       (median_combine=False) or a position dependent PSF is
                       extracted by median combining stars
                       (median_combine=True).
        psfdir:        Path to Hubble PSF. This is necessary if median_combine
                       is set to False.
        RA/DEC:        Coordinates of the galaxy in degrees. This is used to
                       select stars from the neighbourhood if median_combine is
                       set to True.
        GALFIT_tmpdir: Path to directory where temporary files for GALFIT are
                       saved. GALFIT is only run if median_combine is set to
                       True and all the temporary files will be deleted after
                       they have been used.
        save_psf:      True if PSF should be saved for each image.
    """
    # If the PSF used for the fake AGN has to be saved (save_psf==True):
    psf_save_path = conf.run_case+'/psf_used/'
    pixel_scale = 0.06
    fwhm = 0.18
    if median_combine:
        path_to_field = conf.run_case+"/hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits"
        if not os.path.isdir(GALFIT_tmpdir):
            os.makedirs(GALFIT_tmpdir)
        table = astrotable.Table.read(santinidir)
        df = table.to_pandas()
        catalog = df[['RAdeg', 'DECdeg', 'Hmag', 'H_SNR', 'Class_star_1']]
        field_hdu = fits.open(path_to_field)[0]
        w = wcs.WCS(field_hdu.header)
        pixel_coordinates = w.wcs_world2pix(RA, DEC, 0, ra_dec_order=True)
        pixel_y = pixel_coordinates[1]
        pixel_x = pixel_coordinates[0]

        # Filter out hihg SNR stars from the neighborhood of the galaxy.
        star_mask = df['Class_star_1'] > 0.9
        df_stars = df[star_mask]
        SNR_mask = df_stars['H_SNR'] > 5.0
        df_bright = df_stars[SNR_mask]
        numpy_ra = np.array(df_bright['RAdeg'])
        numpy_dec = np.array(df_bright['DECdeg'])
        [x_crds, y_crds] = w.wcs_world2pix(numpy_ra, numpy_dec, 0,
                                           ra_dec_order=True)
        neighbor_mask_size = 4000
        neighbor_mask = np.array(np.ones(len(numpy_ra), dtype=bool))
        for i in range(0, len(numpy_ra)):
            if np.abs(x_crds[i]-pixel_x) > neighbor_mask_size/2 or \
               np.abs(y_crds[i]-pixel_y) > neighbor_mask_size/2:
                neighbor_mask[i]=False
        df_selected = df_bright[neighbor_mask]
        x_coordinates = x_crds[neighbor_mask]
        y_coordinates = y_crds[neighbor_mask]
        h_mags = np.array(df_selected['Hmag'])
        print str(len(x_coordinates)) + ' stars selected.'

        csize = 40
        # get median combined PSF
        psf_unscaled = empirical_PSF(field_hdu.data, x_coordinates,
                                     y_coordinates, csize, fwhm/pixel_scale,
                                     GALFIT_tmpdir, h_mags,
                                     phot_zeropoint=25.9463,
                                     pixel_scale=pixel_scale)

        files_to_delete = glob.glob(GALFIT_tmpdir+'*')
        for f in files_to_delete:
            os.remove(f)

        if psf_unscaled is None:
            return None

    else: # else <==> median_combine==False
        if psfdir=='/mnt/ds3lab/dostark/PSFSTD_WFC3IR_F160W.fits':
            index = random.randint(0, 8)
            print index
            hdu_psf = fits.open(psfdir)[0]
            psf_unscaled = hdu_psf.data[index]
        else:
            psf_unscaled = fits.getdata(psfdir)
        csize = 40

    if median_combine:
        # compute statistics to subtract the background
        psf_centroid = [csize, csize]
        try:
            statmask = photutils.make_source_mask(psf_unscaled, snr=5,
                                                  npixels=5, dilate_size=10)
        except TypeError:
            return None
        bkg_annulus = photutils.CircularAnnulus(psf_centroid, 25, 40)
        bkg_phot_table = photutils.aperture_photometry(psf_unscaled,
                                                       bkg_annulus,
                                                       method='subpixel',
                                                       mask = statmask)
        bkg_mean_per_pixel = bkg_phot_table['aperture_sum'] / bkg_annulus.area()
        src_aperture = photutils.CircularAperture(psf_centroid, 25)
        src_phot_table = photutils.aperture_photometry(psf_unscaled,
                                                       src_aperture,
                                                       method='subpixel')
        flux_photutils = src_phot_table['aperture_sum'] - bkg_mean_per_pixel * \
        src_aperture.area()

        scale_factor = psf_flux / flux_photutils
        psf = scale_factor * (psf_unscaled-bkg_mean_per_pixel)
        #psf = scale_factor * psf_unscaled
    else:
        scale_factor = psf_flux / psf_unscaled.sum()
        psf = scale_factor * psf_unscaled

    if save_psf:
        hdu = fits.PrimaryHDU(psf_unscaled-bkg_mean_per_pixel)
        hdu.writeto(psf_save_path+os.path.basename(origpath), overwrite=True)

    # add scaled PSF to the center of the galaxy
    center = [original.shape[1] // 2, original.shape[0] // 2]
    centroid_galaxy = find_centroid(original)
    centroid_PSF = find_centroid(psf)

    composite_image = np.copy(original)

    gal_x = int(centroid_galaxy[0])
    gal_y = int(centroid_galaxy[1])
    ps_x = int(centroid_PSF[0])
    ps_y = int(centroid_PSF[1])

    # Put the PS on top of the galaxy at its centroid.
    for x in range(0, psf.shape[1]):
        for y in range(0, psf.shape[0]):
            x_rel = gal_x - ps_x + x
            y_rel = gal_y - ps_y + y
            if x_rel >= 0 and y_rel >= 0 and x_rel < original.shape[1] and \
                                                      y_rel < original.shape[0]:
                composite_image[y_rel, x_rel] += psf[y, x]

    return composite_image
