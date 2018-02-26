import time
import argparse
import tensorflow as tf
from astropy.io import fits
from data import *
from model import CGAN
import normalizing

def prepocess_train(img, cond):
    # img = scipy.misc.imresize(img, [conf.adjust_size, conf.adjust_size])
    # cond = scipy.misc.imresize(cond, [conf.adjust_size, conf.adjust_size])
    # h1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.train_size)))
    # w1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.adjust_size)))
    # img = img[h1:h1 + conf.train_size, w1:w1 + conf.train_size]
    # cond = cond[h1:h1 + conf.train_size, w1:w1 + conf.train_size]

    if np.random.random() > 0.5:
        img = np.fliplr(img)
        cond = np.fliplr(cond)

    img = img.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    cond = cond.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    return img, cond


def prepocess_test(img, cond):
    # img = scipy.misc.imresize(img, [conf.train_size, conf.train_size])
    # cond = scipy.misc.imresize(cond, [conf.train_size, conf.train_size])
    img = img.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    cond = cond.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    # img = img/127.5 - 1.
    # cond = cond/127.5 - 1.
    return img, cond


def test(mode):
    data = load_data()
    model = CGAN()

    saver = tf.train.Saver()

    counter = 0
    start_time = time.time()
    out_dir = conf.result_path
    filter_string = conf.filter_
    if not os.path.exists(conf.save_path):
        os.makedirs(conf.save_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    start_epoch = 0
    with tf.Session() as sess:    
        saver.restore(sess, conf.model_path)
        for epoch in xrange(start_epoch, conf.max_epoch):
            if (epoch + 1) % conf.save_per_epoch == 0:
                test_data = data[str(mode)]()
                for img, cond, name in test_data:
                    name = name.replace('-'+filter_string+'.npy', '')
                    pimg, pcond = prepocess_test(img, cond)
                    gen_img = sess.run(model.gen_img, feed_dict={model.image:
                                                                 pimg, 
                                                                 model.cond: 
                                                                 pcond})
                    gen_img = gen_img.reshape(gen_img.shape[1:])
                    Normalizer = normalizing.Normalizer(stretch_type=conf.stretch_type, 
                                            scale_factor=conf.scale_factor,
                                            min_value=conf.pixel_min_value,
                                            max_value=conf.pixel_max_value)
                    fits_recover = Normalizer.unstretch(gen_img[:, :, 0])
                    hdu = fits.PrimaryHDU(fits_recover)
                    save_dir = '%s/epoch_%s/fits_output' % (out_dir, epoch + 1)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    filename = '%s/%s-%s.fits' % (save_dir, name, filter_string)
                    if os.path.exists(filename):
                        os.remove(filename)
                    hdu.writeto(filename)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default="eval")
    args = parser.parse_args()
    mode = args.mode
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf.use_gpu)
    test(mode)
    end_time = time.time()
    print 'inference time: '
    print str(end_time-start_time)