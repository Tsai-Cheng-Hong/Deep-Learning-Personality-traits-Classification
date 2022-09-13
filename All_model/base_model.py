from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, LSTM,\
    BatchNormalization,Bidirectional,TimeDistributed,SpatialDropout1D,\
    GlobalAveragePooling1D,ZeroPadding1D,Conv1D, MaxPooling1D, Embedding,Conv2D,MaxPooling2D,Concatenate,Multiply,add,LeakyReLU


def MLP(embedding_layer):
    Dense_layer2 = GlobalAveragePooling1D()(embedding_layer)
    # Drop_layer1 = Dropout(0.2)(Flatten_layer)
    # Dense_layer1 = Dense(64, activation='relu')(Drop_layer1)
    # Dense_layer2 = Dense(32, activation='relu')(Dense_layer1)
    output1 = Dense(1, activation='sigmoid')(Dense_layer2)
    output2 = Dense(1, activation='sigmoid')(Dense_layer2)
    output3 = Dense(1, activation='sigmoid')(Dense_layer2)
    output4 = Dense(1, activation='sigmoid')(Dense_layer2)
    output5 = Dense(1, activation='sigmoid')(Dense_layer2)
    return output1,output2,output3,output4,output5

def CNN(embedding_layer):
    Conv_layer1 = Conv1D(200, 3, padding='valid', activation='relu', strides=1)(embedding_layer)
    Maxpooling_layer1 = MaxPooling1D(pool_size=2,strides=1)(Conv_layer1)
    Flatten_layer = Flatten()(Maxpooling_layer1)
    Drop_layer1 = Dropout(0.2)(Flatten_layer)
    Dense_layer1 = Dense(64, activation='relu')(Drop_layer1)
    Dense_layer2 = Dense(32, activation='relu')(Dense_layer1)
    output1 = Dense(1, activation='sigmoid')(Dense_layer2)
    output2 = Dense(1, activation='sigmoid')(Dense_layer2)
    output3 = Dense(1, activation='sigmoid')(Dense_layer2)
    output4 = Dense(1, activation='sigmoid')(Dense_layer2)
    output5 = Dense(1, activation='sigmoid')(Dense_layer2)
    return output1,output2,output3,output4,output5

def LSTM_base(embedding_layer):
    LSTM_Layer1 = LSTM(128)(embedding_layer)
    Flatten_layer = Flatten()(LSTM_Layer1)
    Drop_layer1 = Dropout(0.2)(Flatten_layer)
    Dense_layer1 = Dense(64, activation='relu')(Drop_layer1)
    Dense_layer2 = Dense(32, activation='relu')(Dense_layer1)
    output1 = Dense(1, activation='sigmoid')(Dense_layer2)
    output2 = Dense(1, activation='sigmoid')(Dense_layer2)
    output3 = Dense(1, activation='sigmoid')(Dense_layer2)
    output4 = Dense(1, activation='sigmoid')(Dense_layer2)
    output5 = Dense(1, activation='sigmoid')(Dense_layer2)
    return output1, output2, output3, output4, output5

def CNN_LSTM(embedding_layer):
    Conv_layer1 = Conv1D(200, 3, padding='valid', activation='relu', strides=1)(embedding_layer)
    Maxpooling_layer1 = MaxPooling1D(pool_size=2,strides=1)(Conv_layer1)
    Drop_layer1 = Dropout(0.2)(Maxpooling_layer1)
    LSTM_Layer1 = LSTM(200)(Drop_layer1)
    Flatten_layer = Flatten()(LSTM_Layer1)
    Drop_layer1 = Dropout(0.2)(Flatten_layer)
    Dense_layer1 = Dense(64, activation='relu',kernel_regularizer='l2')(Drop_layer1)
    Dense_layer2 = Dense(32, activation='relu',kernel_regularizer='l2')(Dense_layer1)
    output1 = Dense(1, activation='sigmoid')(Dense_layer2)
    output2 = Dense(1, activation='sigmoid')(Dense_layer2)
    output3 = Dense(1, activation='sigmoid')(Dense_layer2)
    output4 = Dense(1, activation='sigmoid')(Dense_layer2)
    output5 = Dense(1, activation='sigmoid')(Dense_layer2)
    return output1,output2,output3,output4,output5

def Text_CNN(embedding_layer):
    Conv_layer1 = Conv1D(128, 1, padding='valid', activation='relu', strides=1)(embedding_layer)
    Conv_layer2 = Conv1D(128, 4, padding='valid', activation='relu', strides=1)(embedding_layer)
    Conv_layer3 = Conv1D(128, 9, padding='valid', activation='relu', strides=1)(embedding_layer)
    Maxpooling_layer1 = GlobalAveragePooling1D()(Conv_layer1)
    Maxpooling_layer2 = GlobalAveragePooling1D()(Conv_layer2)
    Maxpooling_layer3 = GlobalAveragePooling1D()(Conv_layer3)
    Concatenate_layer1 = Concatenate()([Maxpooling_layer1,Maxpooling_layer2])
    Concatenate_layer2 = Concatenate()([Concatenate_layer1,Maxpooling_layer3])
    Flatten_layer = Flatten()(Concatenate_layer2)
    Drop_layer1 = Dropout(0.2)(Flatten_layer)
    # Dense_layer1 = Dense(64, activation='relu')(Drop_layer1)
    # Dense_layer2 = Dense(32, activation='relu')(Dense_layer1)
    Dense_layer1 = Dense(64, activation='tanh')(Drop_layer1)
    Dense_layer2 = Dense(32, activation='tanh')(Dense_layer1)
    # Dense_layer1 = Dense(64)(Drop_layer1)
    # Dense_layer1 = LeakyReLU()(Dense_layer1)
    # Dense_layer2 = Dense(32)(Dense_layer1)
    # Dense_layer2 = LeakyReLU()(Dense_layer2)
    output1 = Dense(1, activation='sigmoid')(Dense_layer2)
    output2 = Dense(1, activation='sigmoid')(Dense_layer2)
    output3 = Dense(1, activation='sigmoid')(Dense_layer2)
    output4 = Dense(1, activation='sigmoid')(Dense_layer2)
    output5 = Dense(1, activation='sigmoid')(Dense_layer2)
    return output1,output2,output3,output4,output5

def Bi_LSTM(embedding_layer):
    Bi_LSTM_Layer1 = Bidirectional(LSTM(200,return_sequences=False, dropout=0.2, recurrent_dropout=0.2,activation = 'tanh'))(embedding_layer)
    Flatten_layer = Flatten()(Bi_LSTM_Layer1)
    Drop_layer1 = Dropout(0.2)(Flatten_layer)
    Dense_layer1 = Dense(64, activation='relu')(Drop_layer1)
    Dense_layer2 = Dense(32, activation='relu')(Dense_layer1)
    output1 = Dense(1, activation='sigmoid')(Dense_layer2)
    output2 = Dense(1, activation='sigmoid')(Dense_layer2)
    output3 = Dense(1, activation='sigmoid')(Dense_layer2)
    output4 = Dense(1, activation='sigmoid')(Dense_layer2)
    output5 = Dense(1, activation='sigmoid')(Dense_layer2)
    return output1, output2, output3, output4, output5

