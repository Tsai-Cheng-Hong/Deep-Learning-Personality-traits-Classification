from tensorflow.python.keras import backend as K

def f1(y_true,y_pred):
    def recall(y_true,y_pred):
        true_pos = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
        possible_pos = K.sum(K.round(K.clip(y_true,0,1)))
        recall = true_pos / (possible_pos + K.epsilon())
        return recall
    def precision(y_true,y_pred):
        true_pos = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
        possible_pos = K.sum(K.round(K.clip(y_pred,0,1)))
        precision = true_pos / (possible_pos + K.epsilon())
        return precision
    precision = precision(y_true,y_pred)
    recall = recall(y_true,y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
