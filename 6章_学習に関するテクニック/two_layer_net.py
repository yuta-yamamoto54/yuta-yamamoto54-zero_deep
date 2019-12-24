import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
#順番が保持された辞書
from collections import OrderedDict
from layer import *
    
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        #input_size*hidden_size
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        #hidden_size*1
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
        #レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        #最後の出力層への活性化関数にはソフトマックス、損失関数には交差エントロピーを用いる
        #ワンセットで誤差逆伝搬がきれいな形に
        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self, x):
        #辞書型のlayersからlayerを順序に取り出す
        for layer in self.layers.values():
            #それぞれのlayerの順伝搬を呼び出す
            x = layer.forward(x)
            
        return x
    
    # x:入力データ, t:教師データ
    #損失関数の出力を返す
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        #最も出力値の高いインデックスをデータごとに取り出す
        y = np.argmax(y, axis=1)
        #もしtがone-hot表現ならラベル表現に直す
        if t.ndim != 1 : t = np.argmax(t, axis=1)
            
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        #gradient内に存在
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        #レイヤを逆にして取り出す
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 設定　入力層における偏微分された値をレイヤーから取得
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads