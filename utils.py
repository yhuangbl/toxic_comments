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


batch_size = 128
epochs = 50

# utils functions
def run_5fold(x_train, y_train, x_test, early_stop_mon, get_model, **kwargs):
    kf = KFold(len(x_train), n_folds=5)
    loss_scores = []
    acc_scores = []
    roc_scores = []
    if early_stop_mon == "auc_roc": 
        earlystop = EarlyStopping(monitor="val_auc_roc", min_delta=0, patience=1, verbose=1, mode="max")
    elif early_stop_mon == "loss":
        earlystop = EarlyStopping(monitor="val_loss", min_delta=0, patience=1, verbose=1, mode="auto")
    else:
        print ("No valid early stopping method")
        return -1
    
    y_preds = []
    for train, test in kf:
        model = get_model(**kwargs)
        hist = model.fit(x_train[train], y_train[train], 
                         batch_size=batch_size, epochs=epochs, verbose=2, 
                         validation_data=(x_train[test], y_train[test]),
                         callbacks=[earlystop])
        
        val_loss = hist.history["val_loss"][-1]
        val_acc = hist.history["val_acc"][-1]
        val_auc_roc = hist.history["val_auc_roc"][-1]
        print ("val loss: {}".format(val_loss))
        print ("val acc: {}".format(val_acc))
        print ("val auc roc: {}\n".format(val_auc_roc))
        loss_scores.append(val_loss)
        acc_scores.append(val_acc)
        roc_scores.append(val_auc_roc)
        
        y_pred = model.predict(x_test, batch_size=batch_size)
        y_preds.append(y_pred)

        # release gpu memory
        K.clear_session()
    
    avg_val_loss = "loss: {} (+/- {})".format(np.mean(loss_scores), np.std(loss_scores))
    avg_val_acc = "acc: {} (+/- {})".format(np.mean(acc_scores), np.std(acc_scores))
    avg_roc_auc = "roc_auc: {} (+/- {})\n\n".format(np.mean(roc_scores), np.std(roc_scores))
    
    result = {"avg_val_loss": avg_val_loss, "avg_val_acc": avg_val_acc, "avg_roc_auc": avg_roc_auc, "pred": y_preds}
    return result

def run_5fold_2channels(x_trains, y_train, x_test, early_stop_mon, get_model, **kwargs):
    # x_train1 and x_train2 are the same
    x_train1 = x_trains[0]
    x_train2 = x_trains[1]
    kf1 = KFold(len(x_train1), n_folds=5)
    loss_scores = []
    acc_scores = []
    roc_scores = []
    if early_stop_mon == "auc_roc": 
        earlystop = EarlyStopping(monitor="val_auc_roc", min_delta=0, patience=1, verbose=1, mode="max")
    elif early_stop_mon == "loss":
        earlystop = EarlyStopping(monitor="val_loss", min_delta=0, patience=1, verbose=1, mode="auto")
    else:
        print ("No valid early stopping method")
        return -1
    
    y_preds = []
    for train, test in kf1:
        model = get_model(**kwargs)
        hist = model.fit([x_train1[train], x_train2[train]], y_train[train],
                         batch_size=batch_size, epochs=epochs, verbose=2,
                         validation_data=([x_train1[test], x_train2[test]], y_train[test]),
                         callbacks=[earlystop])
        
        val_loss = hist.history["val_loss"][-1]
        val_acc = hist.history["val_acc"][-1]
        val_auc_roc = hist.history["val_auc_roc"][-1]
        print ("val loss: {}".format(val_loss))
        print ("val acc: {}".format(val_acc))
        print ("val auc roc: {}\n".format(val_auc_roc))
        loss_scores.append(val_loss)
        acc_scores.append(val_acc)
        roc_scores.append(val_auc_roc)        
        
        y_pred = model.predict([x_test, x_test], batch_size=batch_size)
        y_preds.append(y_pred)
        
        # release gpu memory
        K.clear_session()
        
    avg_val_loss = "loss: {} (+/- {})".format(np.mean(loss_scores), np.std(loss_scores))
    avg_val_acc = "acc: {} (+/- {})".format(np.mean(acc_scores), np.std(acc_scores))
    avg_roc_auc = "roc_auc: {} (+/- {})\n\n".format(np.mean(roc_scores), np.std(roc_scores))
    
    result = {"avg_val_loss": avg_val_loss, "avg_val_acc": avg_val_acc, "avg_roc_auc": avg_roc_auc, "pred": y_preds}
    return result

def average_elementwise(ys):
    y = np.mean(ys, axis=0)
    return y

def write_prediction(ys, submission, output_file):
    y_prediction = average_elementwise(ys)
    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_prediction
    submission.to_csv(output_file, index=False)

def eval_results(results):
    for key in results:
        result = results[key]
        print ("key: {}".format(key))
        print (result["avg_val_loss"])
        print (result["avg_val_acc"])
        print (result["avg_roc_auc"])
        print ()