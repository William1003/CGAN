from utils import *
from ops import *
import tensorflow as tf


class CGAN():
    def __init__(self, sess, epoch, batch_size, z_dim):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size

        # 噪声的维数
        self.z_dim = z_dim

        # 输入输出图片的维数
        self.input_height = 28
        self.input_weight = 28
        self.output_height = 28
        self.output_weight = 28

        # 标签数->0~9
        self.y_dim = 10

        # 色域? 图片(28, 28, 1)?
        self.c_dim = 1

        # train_parameter
        self.learning_rate = 0.0002
        # 这是什么
        self.beta1 = 0.5

        # test

        # 测试时, 保存生成图片的数量
        self.sample_num = 64

        # 加载数据集
        self.data_X, self.data_y = load_mnist()

        # 计算每次个epoch有多少个batch
        self.num_batch = len(self.data_X) // self.batch_size

    # 判别器
    def discriminator(self, x, y, is_training=True, reuse=False):
        # 变量空间
        with tf.variable_scope('discriminator', reuse=reuse):

            # 将图片与标签合并
            y = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(x, y)

            net = tf.nn.leaky_relu(conv2d(x, 64, 4, 4, 24, 2, name='d_conv1'))
            net = tf.nn.leaky_relu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            # -1->把net展平
            net = tf.reshape(net, [self.batch_size, -1])
            net = tf.nn.leaky_relu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            out_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit, net

    # 生成器
    def generator(self, z, y, is_training=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):

            # 将噪声与标签连接
            z = tf.concat([z, y], 1)

            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

            return out

    # 建立模型->判别器 + 生成器
    def bulid_model(self):

        image_dims = [self.input_height, self.input_weight, self.c_dim]
        bs = self.batch_size

        '''输入图片'''
        # 真实图片
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # 标签
        self.y = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')

        # 输入的噪声
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        '''损失函数'''
        # 真实图片输入后得到的输出
        D_real, D_real_logits, _ = self.discriminator(self.batch_size, self.y, is_training=True, reuse=False)

        # 生成器生成假图片, 输出判别器, 得到输出
        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, self.y, is_training=True, reuse=True)

        # 使用交叉熵作为损失函数
        # 真数据的损失
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))

        # 假数据的损失
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        # 判别器的损失
        self.d_loss = d_loss_real + d_loss_fake

        # 生成器的损失
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake))
        )

        '''训练部分'''
        # 将参数划分给D和G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # 参数优化?
        # tf.control_dependencies保证按序执行?
        # tf.get_collection返回key的列表
        # tf.GraphKeys.UPDATE_OPS 待优化的参数?
        # 主要是batch_norm会出错
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdadeltaOptimizer(self.learning_rate, beta1 = self.beta1)
