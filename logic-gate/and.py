import numpy
import chainer
import chainer.optimizers

DATA_SIZE = 1000
BATCH_SIZE = 400

class SmallClassificationModel(chainer.FunctionSet):
    def __init__(self):
        super(SmallClassificationModel, self).__init__(
            fc1 = chainer.functions.Linear(2, 2)
        )

    def _forward(self, x):
        h = self.fc1(x)
        return h

    def train(self, x_data, y_data):
        x = chainer.Variable(x_data.reshape(1, 2).astype(numpy.float32), volatile=False)
        y = chainer.Variable(y_data.astype(numpy.int32), volatile=False)
        h = self._forward(x)

        optimizer.zero_grads()
        error = chainer.functions.softmax_cross_entropy(h, y)
        accuracy = chainer.functions.accuracy(h, y)
        error.backward()
        optimizer.update()
        return accuracy.data

model = SmallClassificationModel()
optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model.collect_parameters())

data_and = [
    [numpy.array([0,0]), numpy.array([0])],
    [numpy.array([0,1]), numpy.array([0])],
    [numpy.array([1,0]), numpy.array([0])],
    [numpy.array([1,1]), numpy.array([1])],
]*DATA_SIZE

idx = 1
result = 0
for invec, outvec in data_and:
    result += model.train(invec, outvec)
    if idx % BATCH_SIZE == 0:
        acc = (result / idx) * 100
        print("Accuracy: {}%".format(acc))
    idx += 1

acc = (result / (DATA_SIZE*4)) * 100
print("Accuracy: {}%".format(acc))
