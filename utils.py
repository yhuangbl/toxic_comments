from keras.callbacks import EarlyStopping
from sklearn.cross_validation import KFold
import numpy as np

earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
batch_size = 128
epochs = 100


def run_5fold(x_train, get_model, **kwargs):
    kf = KFold(len(x_train), n_folds=5)
    acc_scores = []
    roc_scores = []

    for train,test in kf:
        model = get_model(**kwargs)
        hist = model.fit(x_train[train], y_train[train], batch_size=512, epochs=2, verbose=1)
        
        # evaluate the model
        scores = model.evaluate(x_train[test], y_train[test], verbose=0)
        acc_score = scores[1] * 100
        roc_score = scores[2] * 100
        print("%s: %.10f%%" % (model.metrics_names[1], acc_score))
        print("%s: %.10f%%" % (model.metrics_names[2], roc_score))
        acc_scores.append(acc_score)
        roc_scores.append(roc_score)
        
    print("%s: %.10f%% (+/- %.10f%%)" % (model.metrics_names[1], np.mean(acc_scores), np.std(acc_scores)))
    print("%s: %.10f%% (+/- %.10f%%)" % (model.metrics_names[2], np.mean(roc_scores), np.std(roc_scores)))