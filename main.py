import tensorflow as tf
from CGAN import CGAN


def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        cgan = CGAN(sess,
                    epoch=100,
                    batch_size=64,
                    z_dim=64)

        cgan.bulid_model()
        cgan.train()
        print('---------Training finished!--------------')



if __name__ == '__main__':
    main()