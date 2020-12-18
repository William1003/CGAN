from utils import *
from ops import *
import tensorflow as tf
import time
import matplotlib.pyplot as plt


class CGAN():
    def __init__(self, sess, epoch, batch_size, z_dim, test_size=200):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size
        self.test_size = test_size
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

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            # -1->把net展平
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
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
        D_real, D_real_logits, _ = self.discriminator(self.inputs, self.y, is_training=True, reuse=False)

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
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)

        '''测试部分'''

        self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)

        '''总结'''
        d_loss_real_sum = tf.summary.scalar('d_loss_real', d_loss_real)
        d_loss_fake_sum = tf.summary.scalar('d_loss_fake', d_loss_fake)
        d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
        g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)

        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):
        # 初始化所有参数
        tf.global_variables_initializer().run()

        # 创建一些噪声, 供测试用
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        self.test_labels = self.data_y[0:self.batch_size]

        start_epoch = 0
        start_batch_id = 0
        counter = 0

        print('--------start training!------------')

        start_time = time.time()

        d_loss_rec = []
        g_loss_rec = []
        acc = []

        for epoch in range(start_epoch, self.epoch):

            # 按batch训练
            for idx in range(start_batch_id, self.num_batch):
                batch_images = self.data_X[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_labels = self.data_y[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # 更新判别器网络
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.inputs: batch_images, self.y: batch_labels,
                                                                  self.z: batch_z})

                # 五个batch, 更新生成器网络
                if np.mod(counter, 5) == 0:
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                           feed_dict={self.y: batch_labels, self.z: batch_z})

                # 打印总结信息
                print('Epoch: {:2d} {:4d}/{:4d} time: {:4.4f}, d_loss: {:8f}, g_loss: {:.8f}' \
                      .format(epoch, idx, self.num_batch, time.time() - start_time, d_loss, g_loss))

                counter += 1

                # 300个batch保存一次生成的照片
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z, self.y: self.test_labels})

                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './results/' + 'train_{:02d}_{:04d}.png'.format(epoch, idx))

            start_batch_id = 0

            # 记录这个epoch的训练loss, acc

            d_loss_rec.append(d_loss)
            g_loss_rec.append(g_loss)
            acc.append(self.test_discriminator(epoch))

        plt.plot(d_loss_rec)
        plt.xticks(range(epoch))
        plt.title('d_loss')
        plt.xlabel('epoch')
        plt.show()

        plt.plot(g_loss_rec)
        plt.xticks(range(epoch))
        plt.title('g_loss')
        plt.xlabel('epoch')
        plt.show()

        plt.plot(acc)
        plt.xticks(range(epoch))
        plt.title('discriminator_accuracy')
        plt.xlabel('epoch')
        plt.show()


    def test_discriminator(self, epoch):

        # 随机取一些real_image与fake_image混合, 看看判别器准确率
        real_X_index = np.random.choice(self.data_X.shape[0], self.batch_size)
        real_X = np.array([self.data_X[i] for i in real_X_index])[:self.batch_size // 2, :, :, :]
        real_Y = np.array([self.data_y[i] for i in real_X_index]).astype(np.float32)[:self.batch_size // 2, :]

        real_Y = np.concatenate([real_Y, real_Y], axis=0)

        # 生成假数据
        z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)).astype(np.float32)
        fake_X = self.generator(z, real_Y, is_training=False, reuse=True)

        half_fake_X = fake_X[:self.batch_size // 2, :, :, :]

        label_real = np.ones(self.batch_size // 2)
        label_fake = np.zeros(self.batch_size // 2)

        mixed_label = np.concatenate([label_real, label_fake]).astype(np.int32)

        mixed_X = tf.concat([real_X, half_fake_X], 0)

        # tot_num_samples = min(self.sample_num, self.batch_size)
        # manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
        # manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
        # save_images(mixed_X[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
        #             './results/' + 'test_epoch{:02d}.png'.format(epoch))
        mixed_X = tf.random_shuffle(mixed_X, seed=233)
        mixed_label = tf.random_shuffle(mixed_label, seed=233)

        out, out_logit, _ = self.discriminator(mixed_X, real_Y, is_training=False, reuse=True)

        one = tf.ones_like(out, dtype=tf.int32)
        zero = tf.zeros_like(out, dtype=tf.int32)
        out = tf.reshape(tf.where(out >= 0.5, one, zero), [-1])
        res = tf.equal(out, mixed_label)

        correct = tf.count_nonzero(res).eval()

        return correct / self.batch_size
