from tensorflow.python.keras.layers import ZeroPadding1D,Conv1D,Multiply
from tensorflow.python.keras import layers

class GCN(layers.Layer):
    def __init__(self, out_dim, kernel_size, kwarg_conv={}, kwargs_gate={}):
        super(GCN, self).__init__()
        self.conv = Conv1D(out_dim,kernel_size,**kwarg_conv)
        self.conv_gate = Conv1D(out_dim,kernel_size,activation='sigmoid',**kwargs_gate)
        self.pad_input = ZeroPadding1D(padding=(kernel_size-1,0))

    def call(self, inputs):
        X = self.pad_input(inputs)
        Y = self.conv(X)
        Z = self.conv_gate(X)
        return Multiply()([Y,Z])
