from bottle import route, post, request, response, template, static_file, run
from json import dumps
import re
import os
import numpy as np
import chainer
import chainer.links as L
from chainer import Variable
from chainer import serializers
from mnist import MnistModel
from mnist_cnn import MnistCNNModel

def scale_results(pred):
    # 結果を0~1の間にスケールする
    results = []
    scaled_res = []
    pred_sum = 0
    for i in pred:
        if i < 0:
            i = 0
        else:
            i = i / 100
            pred_sum +=  i
        scaled_res.append(i)
    for i in scaled_res:
        results.append((i / pred_sum) * 100)
    return results

@route('/')
def index():
    return template('index')

@post('/pred')
def pred():
    gray_img = ((255 - np.array(request.json, dtype=np.float32)) / 255.0).reshape(1, 784)
    # 訓練済みのデータを使ってモデル初期化
    # MLP
    model_mlp = L.Classifier(MnistModel())
    serializers.load_npz('./output/model_final', model_mlp)
    x = Variable(np.asarray([gray_img], dtype=np.float32), volatile='on')
    y = model_mlp.predictor(x)
    pred_mlp = np.argmax(y.data, axis=1)
    pred_all_mlp = y.data.flatten().tolist()

    # CNN
    model_cnn = L.Classifier(MnistCNNModel())
    serializers.load_npz('./output/model_cnn_final', model_cnn)
    x = Variable(np.asarray([gray_img], dtype=np.float32).reshape(1, 1, 28, 28), volatile='on')
    y = model_cnn.predictor(x)
    pred_cnn = np.argmax(y.data, axis=1)
    pred_all_cnn = y.data.flatten().tolist()

    mlp_results = scale_results(pred_all_mlp)
    cnn_results = scale_results(pred_all_cnn)

    response.content_type = 'application/json'
    return {'pred_mlp': "{}".format(pred_mlp[0]), 'pred_all_mlp': dumps(mlp_results), 'pred_cnn': "{}".format(pred_cnn[0]), 'pred_all_cnn': dumps(cnn_results)}

@route('/static/:path#.+#', name='static')
def static(path):
    return static_file(path, root='static')

run(host='localhost', port=8080, debug=True, reloader=True)
