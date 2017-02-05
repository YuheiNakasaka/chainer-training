import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

# モデル定義
class MnistCNNModel(chainer.Chain):
    insize = 28
    def __init__(self):
        super(MnistCNNModel, self).__init__(
            conv1=L.Convolution2D(1, 20, 5), # conv1: (28-5)/1 + 1 = 24
            conv2=L.Convolution2D(20, 50, 5), # conv2: (12-5)/1 + 1 = 8
            fc3=L.Linear(800, 500), # fc3: 4 * 4 * 50 = 800
            fc4=L.Linear(500, 10),
        )
        self.train = True

    def __call__(self, x):
        # pooling: (width + padding*2 - filter_width)/stride + 1
        # if stride = None, then user filter_width. so in this case, filter_width == stride == 2
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2) # pool1: (24-2)/2 + 1 = 12
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2) # pool2: (8-2)/2 + 1 = 4
        h = F.dropout(F.relu(self.fc3(h)), train=self.train) # p=0.5
        return self.fc4(h)

if __name__ == '__main__':
    model = L.Classifier(MnistCNNModel())
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # 訓練/テストデータ設定
    # train, testデータはそれぞれTupleSet。各tupleは(0-1でスケールされた画像の3次元numpy.array, 正解ラベル)。
    train, test = chainer.datasets.get_mnist(ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

    # 訓練
    # ref) http://docs.chainer.org/en/stable/reference/core/training.html
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (100, 'epoch'), out="result")
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.snapshot(), trigger=(100, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    # 訓練結果の保存
    chainer.serializers.save_npz('./output/model_cnn_final', model)
    chainer.serializers.save_npz('./output/optimizer_cnn_final', optimizer)
