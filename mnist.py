import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

# モデル定義
class MnistModel(chainer.Chain):
    def __init__(self):
        super(MnistModel, self).__init__(
        l1 = L.Linear(784, 100),
        l2 = L.Linear(100, 100),
        l3 = L.Linear(100, 10))

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

if __name__ == '__main__':
    model = L.Classifier(MnistModel())
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # 訓練/テストデータ設定
    # train, testデータはそれぞれTupleSet。各tupleは(0-1でスケールされた画像の1次元numpy.array, 正解ラベル)。
    train, test = chainer.datasets.get_mnist()
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
    chainer.serializers.save_npz('./output/model_final', model)
    chainer.serializers.save_npz('./output/optimizer_final', optimizer)
