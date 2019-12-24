import numpy as np
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *

#活性化関数
class Relu:
    def __init__(self):
        self.mask = None 
    
    def forward(self, x):
        #0以下をTrue,0より大きい要素をFalseとして保持する
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
class Sigmoid:
    def sef__init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out #y(1-y)
        
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #損失
        self.y = None    #softmaxの出力
        self.t = None    #教師データ(one-hot vector)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        #batch_sizeで割ることでデータ一つ当たりの誤差が前のレイヤへ伝搬する
        dx = (self.y - self.t) / batch_size
        
        return dx