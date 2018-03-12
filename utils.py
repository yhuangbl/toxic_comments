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

earlystop = EarlyStopping(monitor="val_auc_roc", min_delta=0, patience=5, verbose=1, mode="max")
batch_size = 128
epochs = 50

# utils functions
def run_5fold(x_train, y_train, get_model, **kwargs):
    kf = KFold(len(x_train), n_folds=5)
    acc_scores = []
    roc_scores = []

    for train, test in kf:
        model = get_model(**kwargs)
        hist = model.fit(x_train[train], y_train[train], 
                         batch_size=batch_size, epochs=epochs, 
                         verbose=2, 
                         validation_data=(x_train[test], y_train[test]),
                         callbacks=[earlystop])
        
        # evaluate the model
        scores = model.evaluate(x_train[test], y_train[test], verbose=0)
        acc_score = scores[1] * 100
        roc_score = scores[2] * 100
        print("%s: %.10f%%" % (model.metrics_names[1], acc_score))
        print("%s: %.10f%%" % (model.metrics_names[2], roc_score))
        acc_scores.append(acc_score)
        roc_scores.append(roc_score)
        
        # release gpu memory
        K.clear_session()
        
    print("%s: %.10f%% (+/- %.10f%%)" % ("acc", np.mean(acc_scores), np.std(acc_scores)))
    print("%s: %.10f%% (+/- %.10f%%)" % ("roc_auc", np.mean(roc_scores), np.std(roc_scores)))

def run_5fold_2channels(x_trains, y_train, get_model, **kwargs):
    # x_train1 and x_train2 are the same
    x_train1 = x_trains[0]
    x_train2 = x_trains[1]
    kf1 = KFold(len(x_train1), n_folds=5)
    acc_scores = []
    roc_scores = []

    for train, test in kf1:
        model = get_model(**kwargs)
        hist = model.fit([x_train1[train], x_train2[train]], y_train[train],
                         batch_size=batch_size, epochs=epochs, verbose=2,
                         validation_data=([x_train1[test], x_train2[test]], y_train[test]),
                         callbacks=[earlystop])
        
        # evaluate the model
        scores = model.evaluate([x_train1[test], x_train2[test]], y_train[test], verbose=0)
        acc_score = scores[1] * 100
        roc_score = scores[2] * 100
        print("%s: %.10f%%" % (model.metrics_names[1], acc_score))
        print("%s: %.10f%%" % (model.metrics_names[2], roc_score))
        acc_scores.append(acc_score)
        roc_scores.append(roc_score)
        
        # release gpu memory
        K.clear_session()
        
    print("%s: %.10f%% (+/- %.10f%%)" % ("acc", np.mean(acc_scores), np.std(acc_scores)))
    print("%s: %.10f%% (+/- %.10f%%)" % ("roc_auc", np.mean(roc_scores), np.std(roc_scores)))