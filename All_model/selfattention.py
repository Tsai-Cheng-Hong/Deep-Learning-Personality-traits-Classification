import os
import tensorflow as tf
import tensorflow.python.keras.backend as K
import numpy
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers import Dot
from scipy.stats import spearmanr




class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape",WQ.shape)
        print("WQ ",WQ)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape) #高度跟寬度互換
        # https://blog.csdn.net/weixin_42078618/article/details/99050835 詳細解析
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (self.output_dim**0.5) #dot product

        distance_layer = Dot(axes=1, normalize=True)  # cosine proximity
        prediction = distance_layer([WQ, WK])
        print("cosine similarity : ", prediction)


        test = tf.add(QK,prediction) #相加
        QK = tf.divide(test,2) #除2，因為2項


        QK = K.softmax(QK) # 計算出來的Correlation作Softmax，變成百分比

        print("QK.shape",QK.shape)
        print("QK",QK)
        V = K.batch_dot(QK,WV) #最後和V作相乘並輸出

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0],input_shape[1],self.output_dim)