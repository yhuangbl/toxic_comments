from utils import max_features, maxlen, embed_size_fastText, embed_size_glove, embed_size_glove_twitter
from utils import batch_size, epochs, earlystop
from model import get_CNN_model, get_multiChannel_CNN_model, get_dpcnn_model, get_textCNN, get_textRCNN_model
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K

# global variables
submission = pd.read_csv("input/sample_submission.csv")
fastText = pickle.load(open("input/fastText.300d.pickle", "rb"))
glove = pickle.load(open("input/glove.300d.pickle", "rb"))
glove_twitter = pickle.load(open("input/glove.twitter.200d.pickle", "rb"))

train = pickle.load(open("input/x_train.pickle", "rb"))
y = pickle.load(open("input/y_train.pickle", "rb"))
x_test = pickle.load(open("input/x_test.pickle", "rb"))
x_train, x_valid, y_train, y_valid = train_test_split(train, y, test_size = 0.1)

def predict2file(output_file, get_model, **params):
    print ("\n**********{}**********".format(output_file))
    model = get_model(**params)
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=(x_valid, y_valid),
                     verbose=2, callbacks=[earlystop])
    y_pred = model.predict(x_test, batch_size=batch_size)
    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
    submission.to_csv(output_file, index=False)
    # release gpu memory
    K.clear_session()

def predict2file_2channels(output_file, get_model, **params):
    print ("\n**********{}**********".format(output_file))
    model = get_model(**params)
    hist = model.fit([x_train, x_train], y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=([x_valid, x_valid], y_valid),
                     verbose=2, callbacks=[earlystop])
    y_pred = model.predict([x_test, x_test], batch_size=batch_size)
    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
    submission.to_csv(output_file, index=False)
    # release gpu memory
    K.clear_session()

params_fastText = {
    "maxlen": maxlen,
    "max_features": max_features,
    "kernel_size": 9,
    "num_filter": 500,
    "reg": 0.0,
    "embedding_matrix": fastText,
    "embed_size": embed_size_fastText, 
    "dropout": 0.1
}
predict2file("output/CNN_fastText.csv", get_CNN_model, **params_fastText)

params_glove = {
    "maxlen": maxlen,
    "max_features": max_features,
    "kernel_size": 7,
    "num_filter": 500,
    "reg": 0.0,
    "embedding_matrix": glove,
    "embed_size": 300, 
    "dropout": 0.2
}
predict2file("output/CNN_glove.csv", get_CNN_model, **params_glove)

params_dpcnn = {
    "maxlen": maxlen,
    "max_features": max_features,
    "kernel_size": 7,
    "num_filter": 128,
    "reg": 0.0,
    "embedding_matrix": fastText,
    "embed_size": 300,
    "dropout": 0.1
}
predict2file("output/dpcnn.csv", get_dpcnn_model, **params_dpcnn)

params_multi = {
    "maxlen": maxlen,
    "max_features": max_features,
    "embed_size1": embed_size_fastText,
    "embedding_matrix1": fastText,
    "dropout1": 0.1,
    "num_filter1": 500,
    "kernel_size1": 9,
    "reg1": 0,
    "embed_size2": embed_size_glove,
    "embedding_matrix2": glove,
    "dropout2": 0.2,
    "num_filter2": 500,
    "kernel_size2": 7,
    "reg2": 0,
    "dense_units": 768
} 
predict2file_2channels("output/multiCNN.csv", get_multiChannel_CNN_model, **params_multi)