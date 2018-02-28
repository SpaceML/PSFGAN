import time
import argparse
import tensorflow as tf
from astropy.io import fits
from data import *
from model import CGAN

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


def train(evalset):
    data = load_data()
    model = CGAN()

    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

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
        if conf.model_path == "":
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, conf.model_path)
            try:
                log = open(conf.save_path + "/log")
                start_epoch = int(log.readline())
                log.close()
            except:
                pass
        for epoch in xrange(start_epoch, conf.max_epoch):
            train_data = data["train"]()
            for img, cond, _ in train_data:
                img, cond = prepocess_train(img, cond)
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image: img, model.cond: cond})
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image: img, model.cond: cond})
                _, M, flux = sess.run([g_opt, model.g_loss, model.delta],
                                      feed_dict={model.image: img, model.cond: cond})
                counter += 1
                print("Iterate [%d]: time: %4.4f, d_loss: %.8f, g_loss: %.8f, flux: %.8f" \
                      % (counter, time.time() - start_time, m, M, flux))
            if (epoch + 1) % conf.save_per_epoch == 0:
                # save_path = saver.save(sess, conf.data_path + "/checkpoint/" + "model_%d.ckpt" % (epoch+1))
                save_path = saver.save(sess, conf.save_path + "/model.ckpt")
                print("Model at epoch %s saved in file: %s" % (epoch + 1, save_path))

                log = open(conf.save_path + "/log", "w")
                log.write(str(epoch + 1))
                log.close()

                test_data = data[str(evalset)]()
                for img, cond, name in test_data:
                    name = name.replace('-'+filter_string+'.npy', '')
                    pimg, pcond = prepocess_test(img, cond)
                    gen_img = sess.run(model.gen_img, feed_dict={model.image: pimg, model.cond: pcond})
                    gen_img = gen_img.reshape(gen_img.shape[1:])

                    fits_recover = conf.unstretch(gen_img[:, :, 0])
                    hdu = fits.PrimaryHDU(fits_recover)
                    save_dir = '%s/epoch_%s/fits_output' % (out_dir, epoch + 1)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    filename = '%s/%s-%s.fits' % (save_dir, name, filter_string)
                    if os.path.exists(filename):
                        os.remove(filename)
                    hdu.writeto(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset",default="eval")
    args = parser.parse_args()
    evalset = args.evalset
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf.use_gpu)
    train(evalset)
