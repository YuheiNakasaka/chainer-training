from __future__ import print_function
import argparse
import os
from PIL import Image
import numpy as np

import chainer
from chainer import cuda
from chainer import training
from chainer.training import extensions
from chainer import Variable
import chainer.functions as F
import chainer.links as L


def add_noise(h, test, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)

class Generator(chainer.Chain):
    def __init__(self, n_hidden, bottom_width=4, ch=512, wscale=0.02):
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        self.ch = ch
        w = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__(
            l0=L.Linear(self.n_hidden, bottom_width*bottom_width*ch, initialW=w),
            dc1=L.Deconvolution2D(ch, ch//2, 4, 2, 1, initialW=w),
            dc2=L.Deconvolution2D(ch//2, ch//4, 4, 2, 1, initialW=w),
            dc3=L.Deconvolution2D(ch//4, ch//8, 4, 2, 1, initialW=w),
            dc4=L.Deconvolution2D(ch//8, 3, 3, 1, 1, initialW=w),
            bn0=L.BatchNormalization(bottom_width*bottom_width*ch),
            bn1=L.BatchNormalization(ch//2),
            bn2=L.BatchNormalization(ch//4),
            bn3=L.BatchNormalization(ch//8),
        )

    def make_hidden(self, batchsize):
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(np.float32)

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0(self.l0(z), test=test)), (z.data.shape[
                      0], self.ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = F.sigmoid(self.dc4(h))
        return x

class Discriminator(chainer.Chain):

    def __init__(self, bottom_width=4, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__(
            c0_0=L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w),
            c0_1=L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w),
            c1_0=L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w),
            c1_1=L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w),
            c2_0=L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w),
            c2_1=L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w),
            c3_0=L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w),
            l4=L.Linear(bottom_width * bottom_width * ch, 1, initialW=w),
            bn0_1=L.BatchNormalization(ch // 4, use_gamma=False),
            bn1_0=L.BatchNormalization(ch // 4, use_gamma=False),
            bn1_1=L.BatchNormalization(ch // 2, use_gamma=False),
            bn2_0=L.BatchNormalization(ch // 2, use_gamma=False),
            bn2_1=L.BatchNormalization(ch // 1, use_gamma=False),
            bn3_0=L.BatchNormalization(ch // 1, use_gamma=False),
        )

    def __call__(self, x, test=False):
        h = add_noise(x, test=test)
        h = F.leaky_relu(add_noise(self.c0_0(h), test=test))
        h = F.leaky_relu(add_noise(self.bn0_1(
            self.c0_1(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn1_0(
            self.c1_0(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn1_1(
            self.c1_1(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn2_0(
            self.c2_0(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn2_1(
            self.c2_1(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn3_0(
            self.c3_0(h), test=test), test=test))
        return self.l4(h)

class DCGANUpdater(training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = y_fake.data.shape[0]
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)

    def loss_gen(self, gen, y_fake):
        batchsize = y_fake.data.shape[0]
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss':loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)

        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        y_real = dis(x_real, test=False)

        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z, test=False)
        y_fake = dis(x_fake, test=False)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)

def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        x = gen(z, test=True)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 3, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 3))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image

def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    gen = Generator(n_hidden=args.n_hidden)
    dis = Discriminator()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    if args.dataset == '':
        # Load the CIFAR10 dataset if args.dataset is not specified
        train, _ = chainer.datasets.get_cifar10(withlabel=False, scale=255.)
    else:
        all_files = os.listdir(args.dataset)
        image_files = [f for f in all_files if ('png' in f or 'jpg' in f)]
        print('{} contains {} image files'
              .format(args.dataset, len(image_files)))
        train = chainer.datasets\
            .ImageDataset(paths=image_files, root=args.dataset)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Set up a trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen, dis,
            10, 10, args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
