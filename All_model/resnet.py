from tensorflow.keras.layers import BatchNormalization,add,AveragePooling1D,ZeroPadding1D,MaxPooling1D,Dropout,Conv1D,ReLU
from .GCNN import GCN
nb_filter = 200
kernel_size = 3
# def identity_Block(x,nb_filter,kernel_size):
#     y = GCN(nb_filter,kernel_size)(x)
#     z = add([x,y])
#     q = BatchNormalization()(z)
#     return q
#
# def identity_Block2(x,nb_filter,kernel_size):
#     y = GCN(nb_filter,kernel_size)(x)
#     z = GCN(nb_filter,kernel_size)(y)
#     q = add([z,x])
#     w = BatchNormalization()(q)
#     return w

def resnet_15(inpt):
    # conv1
    x = Conv1D(nb_filter,kernel_size)(inpt)
    x = MaxPooling1D(pool_size=3, strides=3, padding='same')(x)
    x = BatchNormalization()(x)
    first_in = Dropout(0.1)(x)

    # conv1_x
    x = GCN(nb_filter,kernel_size)(first_in)
    x = BatchNormalization()(x)
    x = GCN(nb_filter,kernel_size)(x)
    x = BatchNormalization()(x)
    x = GCN(nb_filter,kernel_size)(x)
    x = BatchNormalization()(x)
    x = GCN(nb_filter, kernel_size)(x)
    x = BatchNormalization()(x)
    x = GCN(nb_filter, kernel_size)(x)
    x = BatchNormalization()(x)
    x = add([x,first_in])
    x = BatchNormalization()(x)
    second_in = ReLU()(x)

    # conv2_x
    x = GCN(nb_filter,kernel_size)(second_in)
    x = BatchNormalization()(x)
    x = GCN(nb_filter,kernel_size)(x)
    x = BatchNormalization()(x)
    x = GCN(nb_filter,kernel_size)(x)
    x = BatchNormalization()(x)
    x = GCN(nb_filter, kernel_size)(x)
    x = BatchNormalization()(x)
    x = GCN(nb_filter, kernel_size)(x)
    x = BatchNormalization()(x)
    x = add([x,second_in])
    x = BatchNormalization()(x)
    third_in = ReLU()(x)

    # conv3_x
    x = GCN(nb_filter,kernel_size)(third_in)
    x = BatchNormalization()(x)
    x = GCN(nb_filter,kernel_size)(x)
    x = BatchNormalization()(x)
    x = GCN(nb_filter,kernel_size)(x)
    x = BatchNormalization()(x)
    x = GCN(nb_filter, kernel_size)(x)
    x = BatchNormalization()(x)
    x = GCN(nb_filter, kernel_size)(x)
    x = BatchNormalization()(x)
    x = add([x,third_in])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
