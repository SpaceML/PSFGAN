
class Config:
    # data selection parameters
    redshift = '1.25'
    filter_ = 'h'

    # normalization parameters
    stretch_type = 'asinh'
    scale_factor = 50
    
    # model parameters
    learning_rate = 0.00009
    attention_parameter = 0.05
    # if you are not going to train from the very beginning (or only testing),
    # change this path to the existing model path
    model_path = ''
    beta1 = 0.5
    L1_lambda = 100
    sum_lambda = 0
    
    # directory tree setup
    core_path = "/mnt/ds3lab/dostark"
    run_case = "%s/z_%s/%s-band" % (core_path, redshift, filter_)
    ext = '_test_all_stars'
    stretch_setup = '%s/%s_%s%s' % (run_case, stretch_type, scale_factor, ext)
    sub_config = '%s/lintrain_classic_WGAN_%s/lr_%s' % (stretch_setup, 
                                                        attention_parameter,
                                                        learning_rate)
    
    
    output_path = '%s/GAN_output' % sub_config
    result_path = output_path
    data_path = "%s/npy_input" % stretch_setup
    save_path = "%s/model" % sub_config
    
    if '0.01' in run_case:
        pixel_max_value = 41100
    elif '0.05' in run_case:
        pixel_max_value = 6140
    elif '0.1' in run_case:
        pixel_max_value = 1450
        #pixel_max_value = 483 
    elif '0.2' in run_case:
        pixel_max_value = 1657
        #pixel_max_value = 271.0
    pixel_min_value = -0.1
    
    # contrast distribution
    max_contrast_ratio = 10
    min_contrast_ratio = 0.1
    uniform_logspace = False

    # training parameters
    # specify which GPU should be used in CUDA_VISIBLE_DEVICES
    use_gpu = 0
    start_epoch = 0
    save_per_epoch = 50
    max_epoch = 50
    img_size = 424
    train_size = 424
    img_channel = 1
    conv_channel_base = 64
