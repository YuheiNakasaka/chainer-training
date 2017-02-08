import argparse
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, reporter
from chainer import Link, Chain, ChainList
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from scipy.misc import imsave

class Generator(Chain):
    def __init__(self, z_dim):
        super(Generator, self).__init__(
            l1=L.Deconvolution2D(z_dim, 128, 3, 2, 0),
            bn1=L.BatchNormalization(128),
            l2=L.Deconvolution2D(128, 128, 3, 2, 1),
            bn2=L.BatchNormalization(128),
            l3=L.Deconvolution2D(128, 128, 3, 2, 1),
            bn3=L.BatchNormalization(128),
            l4=L.Deconvolution2D(128, 128, 3, 2, 2),
            bn4=L.BatchNormalization(128),
            l5=L.Deconvolution2D(128, 1, 3, 2, 2, outsize=(28, 28)),
        )
        self.train = True

    def __call__(self, z):
        h = self.bn1(F.relu(self.l1(z)), test=not self.train)
        h = self.bn2(F.relu(self.l2(h)), test=not self.train)
        h = self.bn3(F.relu(self.l3(h)), test=not self.train)
        h = self.bn4(F.relu(self.l4(h)), test=not self.train)
        x = F.sigmoid(self.l5(h))
        return x

class Discriminator(Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            l1=L.Convolution2D(None, 32, 3, 2, 1),
            bn1=L.BatchNormalization(32),
            l2=L.Convolution2D(None, 32, 3, 2, 2),
            bn2=L.BatchNormalization(32),
            l3=L.Convolution2D(None, 32, 3, 2, 1),
            bn3=L.BatchNormalization(32),
            l4=L.Convolution2D(None, 32, 3, 2, 1),
            bn4=L.BatchNormalization(32),
            l5=L.Convolution2D(None, 1, 3, 2, 1),
        )
        self.train = True

    def __call__(self, x):
        h = self.bn1(F.leaky_relu(self.l1(x)), test=not self.train)
        h = self.bn2(F.leaky_relu(self.l2(h)), test=not self.train)
        h = self.bn3(F.leaky_relu(self.l3(h)), test=not self.train)
        h = self.bn4(F.leaky_relu(self.l4(h)), test=not self.train)
        y = self.l5(h)
        return y

class GAN_Updater(training.StandardUpdater):
    def __init__(self, iterator, generator, discriminator, optimizers, converter=convert.concat_examples, device=None, z_dim=2,):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.gen = generator
        self.dis = discriminator
        self._optimizers = optimizers
        self.converter = converter
        self.device = device
        self.iteration = 0
        self.z_dim = z_dim

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        true_x = in_arrays  # mnist (200, 1, 28, 28)

        # create input z as random
        batchsize = true_x.shape[0]
        if self.device == 0:
            z = cuda.cupy.random.normal(size=(batchsize, self.z_dim, 1, 1), dtype=np.float32)
            z = Variable(z)
        else:
            z = np.random.uniform(-1, 1, (batchsize, self.z_dim, 1, 1))
            z = z.astype(dtype=np.float32)
            z = Variable(z)

        # G        -> x1                    ->  y of gen
        #              + -> X -> D -> split
        # Truedata -> x2                    ->  y of true data
        gen_output = self.gen(z) # gen_output (200, 1, 28, 28)
        x = F.concat((gen_output, true_x), 0) # gen_output + true_data = (400, 1, 28, 28)
        dis_output = self.dis(x)
        y_gen, y_data = F.split_axis(dis_output, 2, 0) # 0~1 value (200, 1, 1, 1)

        # DがGの生成物を1(間違い), TrueDataを0(正しい)と判定するように学習させる
        # sigmoid_cross_entropy(x, 0) == softplus(x)
        # sigmoid_cross_entropy(x, 1) == softplus(-x)
        loss_gen = F.sum(F.softplus(-y_gen))
        loss_data = F.sum(F.softplus(y_data))
        loss = (loss_gen + loss_data) / batchsize

        for optimizer in self._optimizers.values():
            optimizer.target.cleargrads()

        loss.backward()

        for optimizer in self._optimizers.values():
            optimizer.update()

        reporter.report({'loss':loss, 'gen/loss':loss_gen / batchsize, 'dis/loss':loss_data / batchsize})

        save_image(gen_output, self.epoch, self.device)

# 生成される200枚のうち15x15の数字画像195枚を1枚にまとめて出力する
def save_image(x_gen, epoch, device):
    if device == 0:
        x_gen_img = cuda.to_cpu(x_gen.data)
    else:
        x_gen_img = x_gen.data

    n = x_gen_img.shape[0] # (200, 1, 28, 28)
    n = n // 15 * 15 # 195
    x_gen_img = x_gen_img[:n] # (195, 1, 28, 28)
    x_gen_img = x_gen_img.reshape(15, -1, 28, 28) # (15, 13, 28, 28)
    x_gen_img = x_gen_img.transpose(1, 2, 0, 3) # (13, 28, 15, 28)
    x_gen_img = x_gen_img.reshape(-1, 15 * 28) # (364, 420)
    imsave("./output/device{}_x_gen_{}.png".format(device, epoch), x_gen_img)

def main():
    parser = argparse.ArgumentParser(description='DCGAN_MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=200, help='Number of the mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of the training epoch')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='If use GPU, then 0.(-1 is CPU)')
    args = parser.parse_args()

    z_dim = 2
    batch_size = args.batchsize
    epoch = args.epoch
    device = args.gpu
    output = "result{}".format(device)

    print("GPU: {}".format(device))
    print("BatchSize: {}".format(batch_size))
    print("Epoch: {}".format(epoch))

    gen = Generator(z_dim)
    dis = Discriminator()
    if device == 0:
        gen.to_gpu()
        dis.to_gpu()

    opt = {'gen': optimizers.Adam(alpha=-0.001, beta1=0.5),
           'dis': optimizers.Adam(alpha=0.001, beta1=0.5)}
    opt['gen'].setup(gen)
    opt['dis'].setup(dis)

    train, test = datasets.get_mnist(withlabel=False, ndim=3)
    train_iter = iterators.SerialIterator(train, batch_size=batch_size)

    updater = GAN_Updater(train_iter, gen, dis, opt, device=device, z_dim=z_dim)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=output)

    trainer.extend(extensions.dump_graph('loss'))
    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'loss', 'loss_gen', 'loss_data']))
    trainer.extend(extensions.ProgressBar(update_interval=100))

    trainer.run()

if __name__ == '__main__':
    main()
