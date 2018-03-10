from keras.callbacks import EarlyStopping
from sklearn.cross_validation import KFold
import numpy as np
from keras import backend as K

# some global variables
max_features = 30000
maxlen = 200
embed_size_fastText = 300
embed_size_glove = 300
embed_size_glove_twitter = 200

earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=0, verbose=1, mode='auto')
batch_size = 128
epochs = 3

# utils functions
def run_5fold(x_train, y_train, batch_size, get_model, **kwargs):
    kf = KFold(len(x_train), n_folds=3)
    acc_scores = []
    roc_scores = []

    for train,test in kf:
        model = get_model(**kwargs)
        hist = model.fit(x_train[train], y_train[train], batch_size=batch_size, epochs=epochs, verbose=2)
        
        # evaluate the model
        scores = model.evaluate(x_train[test], y_train[test], verbose=0)
        acc_score = scores[1] * 100
        roc_score = scores[2] * 100
        print("%s: %.10f%%" % (model.metrics_names[1], acc_score))
        print("%s: %.10f%%" % (model.metrics_names[2], roc_score))
        acc_scores.append(acc_score)
        roc_scores.append(roc_score)
        # release gpu memory
        del model
        K.clear_session()
        
    print("%s: %.10f%% (+/- %.10f%%)" % ("acc", np.mean(acc_scores), np.std(acc_scores)))
    print("%s: %.10f%% (+/- %.10f%%)" % ("roc_auc", np.mean(roc_scores), np.std(roc_scores)))