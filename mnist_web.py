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

@route('/')
def index():
    return template('index')

@post('/pred')
def pred():
    gray_img = ((255 - np.array(request.json, dtype=np.float32)) / 255.0).reshape(1, 784)
    # 訓練済みのデータを使ってモデル初期化
    model = L.Classifier(MnistModel())
    serializers.load_npz('./output/model_final', model)
    x = Variable(np.asarray([gray_img], dtype=np.float32), volatile='on')
    y = model.predictor(x)
    pred = np.argmax(y.data, axis=1)
    pred_all = y.data.flatten().tolist()

    # 結果を0~1の間にスケールする
    results = []
    scaled_res = []
    pred_sum = 0
    for i in pred_all:
        if i < 0:
            i = 0
        else:
            i = i / 100
            pred_sum +=  i
        scaled_res.append(i)
    for i in scaled_res:
        results.append((i / pred_sum) * 100)

    response.content_type = 'application/json'
    return {'pred_value': "{}".format(pred[0]), 'pred_all': dumps(results)}

@route('/static/:path#.+#', name='static')
def static(path):
    return static_file(path, root='static')

run(host='localhost', port=8080, debug=True, reloader=True)
