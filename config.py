import os

class Config:
    ## data selection parameters
    redshift = '0.1'
    filter_ = 'r'

    ## normalization parameters
    stretch_type = 'asinh'
    scale_factor = 50

    ## model parameters
    learning_rate = 0.00009
    attention_parameter = 0.05
    # if you are not going to train from the very beginning (or only testing),
    # change this path to the existing model path (.cpkt file)
    model_path = ''
    beta1 = 0.5
    L1_lambda = 100
    sum_lambda = 0

    ## directory tree setup
    # 1/ Dataset dependant
    # Working directory where the project is stored
    # Default to where this file is stored
    core_path = os.path.dirname(os.path.abspath(__file__))
    run_case = "%s/z_%s/%s-band" % (core_path, redshift, filter_)
    ext = '' #Custom extension to differentiate setups
    # 2/ Precomputation dependant
    stretch_setup = '%s/%s_%s%s' % (run_case, stretch_type, scale_factor, ext)
    # 3/ PSFGAN model dependant
    sub_config = '%s/lintrain_classic_PSFGAN_%s/lr_%s' % (stretch_setup,
                                                        attention_parameter,
                                                        learning_rate)


    output_path = '%s/PSFGAN_output' % sub_config
    result_path = output_path
    data_path = "%s/npy_input" % stretch_setup
    save_path = "%s/model" % sub_config

    ## Datasets dependant value
    # This has been precomputed for SDSS datasets
    if '0.01' in run_case:
        pixel_max_value = 41100
    elif '0.05' in run_case:
        pixel_max_value = 6140
    elif '0.1' in run_case:
        pixel_max_value = 1450
    elif '0.2' in run_case:
        pixel_max_value = 1657
    pixel_min_value = -0.1

    ## contrast distribution for the added PSF
    max_contrast_ratio = 10
    min_contrast_ratio = 0.1
    uniform_logspace = False

    ## training parameters
    # specify which GPU should be used in CUDA_VISIBLE_DEVICES
    use_gpu = 0
    start_epoch = 0
    save_per_epoch = 50
    max_epoch = 50
    img_size = 424
    train_size = 424
    img_channel = 1
    conv_channel_base = 64
