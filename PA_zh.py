import logging
import pandas as pd
import re
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score,train_test_split
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, LSTM,\
    BatchNormalization,Bidirectional,TimeDistributed,SpatialDropout1D,\
    GlobalAveragePooling1D,ZeroPadding1D,Conv1D, MaxPooling1D, Embedding,Conv2D,MaxPooling2D,Concatenate,Multiply,add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers,Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend
from numpy import array,asarray,zeros
import numpy as np
import re
from gensim.models import KeyedVectors
# Here import model
from All_model.attention import Attention
from All_model.transformer import TransformerBlock,TokenAndPositionEmbedding
from All_model.GCNN import GCN
from All_model.base_model import MLP,CNN,LSTM_base,CNN_LSTM,Text_CNN,Bi_LSTM
from All_model.resnet import resnet_15
from Evalutate.f1_score import f1
from All_model.my_model import *
# 這邊記得替換成檔案的路徑
import tensorflow as tf
import time
start_time = time.time()
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import matplotlib.pyplot as plt

# 1.讀檔
#---------------------------------------------------------------------#
toxic_comments = pd.read_csv('./data/mypersonality_zh.csv',encoding='utf-8')
# toxic_comments = pd.read_csv('./data/essays_all.csv',encoding='utf-8')
# toxic_comments = pd.read_csv('./data/friends-personality_full.csv',encoding='utf-8')
# toxic_comments = pd.read_csv('./data/full_dataset.csv',encoding='utf-8')

print("All Data & Title",toxic_comments.shape)
toxic_comments.head()
# filter = toxic_comments["STATUS"] != ""
filter = toxic_comments["STATUS_zh"] != ""
toxic_comments = toxic_comments[filter]
toxic_comments = toxic_comments.dropna()
# print("第168行的內容",toxic_comments["STATUS"][168])
# print("外向性:" + str(toxic_comments["cEXT"][168]))
# print("神經質:" + str(toxic_comments["cNEU"][168]))
# print("親和性:" + str(toxic_comments["cAGR"][168]))
# print("盡責性:" + str(toxic_comments["cCON"][168]))
# print("經驗開放性:" + str(toxic_comments["cOPN"][168]))
toxic_comments_labels = toxic_comments[["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]]
print(toxic_comments_labels.head())
# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 10
# fig_size[1] = 8
# plt.rcParams["figure.figsize"] = fig_size
# toxic_comments_labels.sum(axis=0).plot.bar()
# plt.show()
#---------------------------------------------------------------------#


# 2.創建多標籤
#---------------------------------------------------------------------#
def preprocess_text(sen):
    # Remove punctuations and numbers
    # sentence = re.sub('[^a-zA-Z]', ' ', sen)
    #
    # Single character removal
    # sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    #
    # Removing multiple spaces
    # sentence = re.sub(r'\s+', ' ', sentence)
    # return sentence
    return sen
X = []
# sentences = list(toxic_comments["STATUS"])
sentences = list(toxic_comments["STATUS_zh"])
for sen in sentences:
    X.append(preprocess_text(sen))

y = toxic_comments[["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
all_labels = Counter(y_train)
print("Before undersampling: ",all_labels)
# undersample = RandomUnderSampler(sampling_strategy='majority')
# X_train,y_train = undersample.fit_resample(X_train,y_train)
# print("After undersampling: ",Counter(y_train))
# print("Undersampling finish")

y1_train = y_train[["cEXT"]].values
y1_test =  y_test[["cEXT"]].values

# Second output
y2_train = y_train[["cNEU"]].values
y2_test =  y_test[["cNEU"]].values

# Third output
y3_train = y_train[["cAGR"]].values
y3_test =  y_test[["cAGR"]].values

# Fourth output
y4_train = y_train[["cCON"]].values
y4_test =  y_test[["cCON"]].values

# Fifth output
y5_train = y_train[["cOPN"]].values
y5_test =  y_test[["cOPN"]].values

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
# embedding_size = 100
embedding_size = 300
maxlen = 300

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# ------------------Personality vector begin------------------ #
# pretrain_w2v_model = KeyedVectors.load_word2vec_format('./pretrained_model/personality_vec',binary=False)
# pretrain_w2v_model = KeyedVectors.load_word2vec_format('./pretrained_model/zh_personality',binary=False)
# pretrain_w2v_model = KeyedVectors.load_word2vec_format('./pretrained_model/seg_ch',binary=False)
# print("Personality Corpus load ok")
pretrain_w2v_model2 = KeyedVectors.load_word2vec_format('./pretrained_model/word_vec',binary=False)
print("Wiki Corpus load ok")
# 1
# personality_matrix = np.zeros((len(word_index) + 1,  embedding_size))
# for word, i in word_index.items():
#     if word in pretrain_w2v_model:
#         personality_matrix[i] = np.asarray(pretrain_w2v_model[word],dtype='float32')

# 2
embedding_matrix = np.zeros((len(word_index) + 1,  embedding_size))
for word, i in word_index.items():
    if word in pretrain_w2v_model2:
        embedding_matrix[i] = np.asarray(pretrain_w2v_model2[word],dtype='float32')

# ------------------Personality vector end------------------ #

#---------------------------------------------------------------------#
backend.clear_session()
# 3.Create Model
#---------------------------------------------------------------------#
input_1 = Input(shape=(maxlen,))
# embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], trainable=False)(input_1)
embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], trainable=True)(input_1)
# personality_embedding_layer = Embedding(len(word_index) + 1,
#                                 embedding_size,
#                                 weights=[personality_matrix],
#                                 input_length=maxlen,
#                                 trainable=True)(input_1)
# embedding_layer = CapsNet()(embedding_layer)
output1 ,output2 ,output3 ,output4 ,output5 = MLP(embedding_layer) # MLP
# output1 ,output2 ,output3 ,output4 ,output5 = CNN(embedding_layer) # CNN
# output1 ,output2 ,output3 ,output4 ,output5 = LSTM_base(embedding_layer) # LSTM
# output1 ,output2 ,output3 ,output4 ,output5 =CNN_LSTM(embedding_layer) # CNN_LSTM
# output1 ,output2 ,output3 ,output4 ,output5 =Text_CNN(embedding_layer) # Text_CNN - base
# output1 ,output2 ,output3 ,output4 ,output5 =Text_CNN(personality_embedding_layer) # Text_CNN - our feature
# output1 ,output2 ,output3 ,output4 ,output5 =mine(embedding_layer,personality_embedding_layer) # Text_CNN - mix feature
# output1 ,output2 ,output3 ,output4 ,output5 =mine2(embedding_layer,personality_embedding_layer) # Text_CNN - mix feature
# output1 ,output2 ,output3 ,output4 ,output5 =Bi_LSTM(embedding_layer) # Bi_LSTM

# Position Encoding & Transformers
# position_embedding_layer = embedding_layer
# position_embedding_layer = TokenAndPositionEmbedding(maxlen=maxlen, vocab_size=10000, embed_dim=embedding_size)(input_1)
# transformer_block = TransformerBlock(embed_dim=embedding_size,num_heads=2,ff_dim=32)
# Transformer_layer1 = transformer_block(position_embedding_layer)
# Global_AVG_Pooling_layer1 = GlobalAveragePooling1D()(Transformer_layer1)
# Flatten_layer = Flatten()(Global_AVG_Pooling_layer1)
# Drop_layer1 = Dropout(0.2)(Flatten_layer)
# Dense_layer1 = Dense(64, activation='relu')(Drop_layer1)
# Dense_layer2 = Dense(32, activation='relu')(Dense_layer1)
# output1 = Dense(1, activation='sigmoid')(Dense_layer2)
# output2 = Dense(1, activation='sigmoid')(Dense_layer2)
# output3 = Dense(1, activation='sigmoid')(Dense_layer2)
# output4 = Dense(1, activation='sigmoid')(Dense_layer2)
# output5 = Dense(1, activation='sigmoid')(Dense_layer2)

# Gated_CNN ResNet-34
# Resnet34_layer = resnet_34(embedding_layer)
# Resnet34_layer = BatchNormalization()(Resnet34_layer)
# Flatten_layer = Flatten()(Resnet34_layer)
# Drop_layer1 = Dropout(0.2)(Flatten_layer)
# Dense_layer1 = Dense(64, activation='relu')(Drop_layer1)
# Dense_layer2 = Dense(32, activation='relu')(Dense_layer1)
# output1 = Dense(1, activation='sigmoid')(Dense_layer2)
# output2 = Dense(1, activation='sigmoid')(Dense_layer2)
# output3 = Dense(1, activation='sigmoid')(Dense_layer2)
# output4 = Dense(1, activation='sigmoid')(Dense_layer2)
# output5 = Dense(1, activation='sigmoid')(Dense_layer2)
#---------------------------------------------------------------------#

# 4.Training Model
#---------------------------------------------------------------------#
model = Model(inputs=input_1, outputs=[output1, output2, output3, output4, output5])
earlystop = EarlyStopping(monitor="val_loss",
                            patience=5,
                            verbose=1,
                          restore_best_weights=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

history = model.fit(x=X_train, y=[y1_train, y2_train, y3_train, y4_train, y5_train], batch_size=64, epochs=50, verbose=1, validation_split=0.2,callbacks=[earlystop],class_weight=None)
# history = model.fit(x=X_train, y=[y1_train, y2_train, y3_train, y4_train, y5_train], batch_size=64, epochs=10, verbose=1)

# print("=====================================================================================")
print("Training:")
# print("外向性E Loss : %.4f"%min(history.history['dense_2_loss']),"Acc : %.4f"%max(history.history['dense_2_acc']),"F1 score : %.4f"%max(history.history['dense_2_f1']))
# print("神經質N Loss : %.4f"%min(history.history['dense_3_loss']),"Acc : %.4f"%max(history.history['dense_3_acc']),"F1 score : %.4f"%max(history.history['dense_3_f1']))
# print("親和性A Loss : %.4f"%min(history.history['dense_4_loss']),"Acc : %.4f"%max(history.history['dense_4_acc']),"F1 score : %.4f"%max(history.history['dense_4_f1']))
# print("盡責性C Loss : %.4f"%min(history.history['dense_5_loss']),"Acc : %.4f"%max(history.history['dense_5_acc']),"F1 score : %.4f"%max(history.history['dense_5_f1']))
# print("開放性O Loss : %.4f"%min(history.history['dense_6_loss']),"Acc : %.4f"%max(history.history['dense_6_acc']),"F1 score : %.4f"%max(history.history['dense_6_f1']))

print("外向性E Loss : %.4f"%min(history.history['dense_loss']),"Acc : %.4f"%max(history.history['dense_acc']),"F1 score : %.4f"%max(history.history['dense_f1']))
print("神經質N Loss : %.4f"%min(history.history['dense_1_loss']),"Acc : %.4f"%max(history.history['dense_1_acc']),"F1 score : %.4f"%max(history.history['dense_1_f1']))
print("親和性A Loss : %.4f"%min(history.history['dense_2_loss']),"Acc : %.4f"%max(history.history['dense_2_acc']),"F1 score : %.4f"%max(history.history['dense_2_f1']))
print("盡責性C Loss : %.4f"%min(history.history['dense_3_loss']),"Acc : %.4f"%max(history.history['dense_3_acc']),"F1 score : %.4f"%max(history.history['dense_3_f1']))
print("開放性O Loss : %.4f"%min(history.history['dense_4_loss']),"Acc : %.4f"%max(history.history['dense_4_acc']),"F1 score : %.4f"%max(history.history['dense_4_f1']))


# Transformer
# print("外向性E Loss : %.4f"%min(history.history['dense_4_loss']),"Acc : %.4f"%max(history.history['dense_4_acc']))
# print("神經質N Loss : %.4f"%min(history.history['dense_5_loss']),"Acc : %.4f"%max(history.history['dense_5_acc']))
# print("親和性A Loss : %.4f"%min(history.history['dense_6_loss']),"Acc : %.4f"%max(history.history['dense_6_acc']))
# print("盡責性C Loss : %.4f"%min(history.history['dense_7_loss']),"Acc : %.4f"%max(history.history['dense_7_acc']))
# print("開放性O Loss : %.4f"%min(history.history['dense_8_loss']),"Acc : %.4f"%max(history.history['dense_8_acc']))

print("=====================================================================================")
print("Validation:")
# print("外向性E Loss : %.4f"%min(history.history['val_dense_2_loss']),"Acc : %.4f"%max(history.history['val_dense_2_acc']),"F1 score : %.4f"%max(history.history['val_dense_2_f1']))
# print("神經質N Loss : %.4f"%min(history.history['val_dense_3_loss']),"Acc : %.4f"%max(history.history['val_dense_3_acc']),"F1 score : %.4f"%max(history.history['val_dense_3_f1']))
# print("親和性A Loss : %.4f"%min(history.history['val_dense_4_loss']),"Acc : %.4f"%max(history.history['val_dense_4_acc']),"F1 score : %.4f"%max(history.history['val_dense_4_f1']))
# print("盡責性C Loss : %.4f"%min(history.history['val_dense_5_loss']),"Acc : %.4f"%max(history.history['val_dense_5_acc']),"F1 score : %.4f"%max(history.history['val_dense_5_f1']))
# print("開放性O Loss : %.4f"%min(history.history['val_dense_6_loss']),"Acc : %.4f"%max(history.history['val_dense_6_acc']),"F1 score : %.4f"%max(history.history['val_dense_6_f1']))

print("外向性E Loss : %.4f"%min(history.history['val_dense_loss']),"Acc : %.4f"%max(history.history['val_dense_acc']),"F1 score : %.4f"%max(history.history['val_dense_f1']))
print("神經質N Loss : %.4f"%min(history.history['val_dense_1_loss']),"Acc : %.4f"%max(history.history['val_dense_1_acc']),"F1 score : %.4f"%max(history.history['val_dense_1_f1']))
print("親和性A Loss : %.4f"%min(history.history['val_dense_2_loss']),"Acc : %.4f"%max(history.history['val_dense_2_acc']),"F1 score : %.4f"%max(history.history['val_dense_2_f1']))
print("盡責性C Loss : %.4f"%min(history.history['val_dense_3_loss']),"Acc : %.4f"%max(history.history['val_dense_3_acc']),"F1 score : %.4f"%max(history.history['val_dense_3_f1']))
print("開放性O Loss : %.4f"%min(history.history['val_dense_4_loss']),"Acc : %.4f"%max(history.history['val_dense_4_acc']),"F1 score : %.4f"%max(history.history['val_dense_4_f1']))

# Transformer
# print("外向性E Loss : %.4f"%min(history.history['val_dense_4_loss']),"Acc : %.4f"%max(history.history['val_dense_4_acc']))
# print("神經質N Loss : %.4f"%min(history.history['val_dense_5_loss']),"Acc : %.4f"%max(history.history['val_dense_5_acc']))
# print("親和性A Loss : %.4f"%min(history.history['val_dense_6_loss']),"Acc : %.4f"%max(history.history['val_dense_6_acc']))
# print("盡責性C Loss : %.4f"%min(history.history['val_dense_7_loss']),"Acc : %.4f"%max(history.history['val_dense_7_acc']))
# print("開放性O Loss : %.4f"%min(history.history['val_dense_8_loss']),"Acc : %.4f"%max(history.history['val_dense_8_acc']))

print("=====================================================================================")
print("Test:")
score = model.evaluate(x=X_test, y=[y1_test, y2_test, y3_test, y4_test, y5_test], verbose=1)
print("外向性E Loss : %.4f"%score[1],"Acc : %.4f"%score[6] ,"F1 score : %.4f"%score[7])
print("神經質N Loss : %.4f"%score[2],"Acc : %.4f"%score[8] ,"F1 score : %.4f"%score[9])
print("親和性A Loss : %.4f"%score[3],"Acc : %.4f"%score[10],"F1 score : %.4f"%score[11])
print("盡責性C Loss : %.4f"%score[4],"Acc : %.4f"%score[12],"F1 score : %.4f"%score[13])
print("開放性O Loss : %.4f"%score[5],"Acc : %.4f"%score[14],"F1 score : %.4f"%score[15])

# model.save('./save/model.h5')





# Text-CNN & Bi-LSTM
# Conv_layer1 = Conv1D(128, 3, padding='valid', activation='relu', strides=1)(embedding_layer)
# Maxpooling_layer1 = MaxPooling1D()(Conv_layer1)
# Conv_layer2 = Conv1D(64, 3, padding='valid', activation='relu', strides=1)(embedding_layer)
# Maxpooling_layer2 = MaxPooling1D()(Conv_layer2)
# Conv_layer3 = Conv1D(32, 3, padding='valid', activation='relu', strides=1)(embedding_layer)
# Maxpooling_layer3 = MaxPooling1D()(Conv_layer3)
# Concatenate_layer1 = Concatenate()([Maxpooling_layer1,Maxpooling_layer2])
# Concatenate_layer2 = Concatenate()([Concatenate_layer1,Maxpooling_layer3])
# Drop_layer1 = Dropout(0.2)(Concatenate_layer2)
# Bi_LSTM_Layer1 = Bidirectional(LSTM(200,return_sequences=False, dropout=0.2, recurrent_dropout=0.2,activation = 'tanh'))(Drop_layer1)
# Flatten_layer = Flatten()(Bi_LSTM_Layer1)
# Drop_layer2 = Dropout(0.2)(Flatten_layer)
# Dense_layer1 = Dense(64, activation='relu')(Drop_layer2)
# Dense_layer2 = Dense(32, activation='relu')(Dense_layer1)
# output1 = Dense(1, activation='sigmoid')(Dense_layer2)
# output2 = Dense(1, activation='sigmoid')(Dense_layer2)
# output3 = Dense(1, activation='sigmoid')(Dense_layer2)
# output4 = Dense(1, activation='sigmoid')(Dense_layer2)
# output5 = Dense(1, activation='sigmoid')(Dense_layer2)

# Attention & Bi-LSTM
# Bi_LSTM_Layer1 = Bidirectional(LSTM(200,return_sequences=True, dropout=0.2, recurrent_dropout=0.2,activation = 'tanh'))(embedding_layer)
# Attention_layer1 = Attention(maxlen)(Bi_LSTM_Layer1)
# Flatten_layer = Flatten()(Attention_layer1)
# Drop_layer1 = Dropout(0.2)(Flatten_layer)
# Dense_layer1 = Dense(64, activation='relu')(Drop_layer1)
# Dense_layer2 = Dense(32, activation='relu')(Dense_layer1)
# output1 = Dense(1, activation='sigmoid')(Dense_layer2)
# output2 = Dense(1, activation='sigmoid')(Dense_layer2)
# output3 = Dense(1, activation='sigmoid')(Dense_layer2)
# output4 = Dense(1, activation='sigmoid')(Dense_layer2)
# output5 = Dense(1, activation='sigmoid')(Dense_layer2)



end_time = time.time()
total = end_time - start_time
print('total cost time : %.2f' %total)