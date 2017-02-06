import numpy as np
import chainer
import chainer.links as L
from chainer import Variable
from chainer import serializers
from mnist import MnistModel

train, test = chainer.datasets.get_mnist()

# 訓練済みのデータを使ってモデル初期化
model = L.Classifier(MnistModel())
serializers.load_npz('./output/model_final', model)
x, t = test[1]

x = Variable(x.reshape(1, 784), volatile='on')
y = model.predictor(x)
pred = np.argmax(y.data, axis=1)
print(y.data.flatten().tolist())
print("Acc: {}, Pred: {}".format(t, pred))
