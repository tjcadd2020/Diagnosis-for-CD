from imblearn.over_sampling import SMOTE
from math import asin, sqrt
import numpy as np
import pandas as pd


from scipy.stats import norm, pearsonr, spearmanr
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, accuracy_score, auc, recall_score, precision_score, average_precision_score, roc_auc_score, f1_score
from sklearn.metrics import balanced_accuracy_score

#model construction
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


class machine_learning:
    
    def __init__(self):
        self.Method = {'Logistic(l1)':LogisticRegression(penalty='l1', random_state=RANDOM_SEED, solver='liblinear', class_weight='balanced'),
                  'Logistic(l2)':LogisticRegression(penalty='l2', random_state=RANDOM_SEED, solver='liblinear', class_weight='balanced'),
                  'DecisionTree':DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_SEED),
                  'RandomForest':RandomForestClassifier(oob_score=True, class_weight='balanced'),
                  'GradientBoost':GradientBoostingClassifier(random_state=RANDOM_SEED)
                  }
        

    def scoring(self,clf, x, y): 
        proba = clf.predict(x) 
        pred = np.array([1 if x>0.5 else 0 for x in proba])
        TP = ((pred==1) & (y==1)).sum()
        FP = ((pred==1) & (y==0)).sum()
        TN = ((pred==0) & (y==0)).sum()
        FN = ((pred==0) & (y==1)).sum()
        sen = TP/ float(TP + FN)
        spe = TN / float(FP + TN)
        recall = recall_score(y, pred)
        precision = precision_score(y, pred)
        accuracy = accuracy_score(y, pred)
        auc = roc_auc_score(y, proba)
        f1 = f1_score(y, pred)
        return [pred, proba, sen, spe, recall, precision,accuracy, f1, auc]
            

   
    def cv_train(self,X, Y, clf, params, k_fold, group_list): 
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto') 
        
        k, tprs, aucs, mean_fpr, plot_lines = 0, [], [], np.linspace(0, 1, 100), []
        sens, spes = [],[]
        scores = np.zeros([k_fold, 7])

  
        for train_index, test_index in spt:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            posi=list(Y_train).count(1)
            nega=list(Y_train).count(0)
            total=posi+nega
            
            #加入class_weight
            weight_for_0 = (1 / nega)*(total)/2.0 
            weight_for_1 = (1 / posi)*(total)/2.0
            class_weight = {0: weight_for_0, 1: weight_for_1}
            

            clf = load_model('origin.h5') # origin model
            
            # 设置模型参数
            clf.fit(X_train, Y_train,batch_size=params['batch_size'], epochs=params['nb_epoch'],class_weight = class_weight, validation_data=(X_test, Y_test), shuffle=False, callbacks=[reduce_lr])  ### clf



            pred, proba, sen, spe, recall, precision,accuracy, f1, auc_score = self.scoring(clf, X_test, Y_test)

            aucs.append(auc_score)
            sens.append(sen)
            spes.append(spe)
            scores[k] = [accuracy, precision, recall, f1,sen,spe, auc_score]
            
            fpr, tpr, thresholds = roc_curve(Y_test, proba) # fpr,tpr 
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            
            plot_lines.append([fpr, tpr, 'ROC Fold %d(AUC = %0.2f)' %(k+1, auc_score)])

            k += 1


        scores = pd.DataFrame(scores, index=['Fold'+str(i+1) for i in range(k_fold)], columns=['accuracy', 'precision', 'recall', 'f1','sensitivity','specificity', 'auc'])

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        mean_sen = np.mean(sens)
        mean_spe = np.mean(spes)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        plot_data = (plot_lines, mean_auc, std_auc, mean_tpr, std_tpr, tprs_upper,tprs_lower, mean_fpr)

        return scores, aucs, plot_data, mean_sen, mean_spe

    def test(self,clf, X_test, Y_test): 

        tprs, aucs, mean_fpr =  [], [], np.linspace(0, 1, 100)
        pred, proba, sen, spe, recall, precision,accuracy, f1, auc_score = self.scoring(clf, X_test, Y_test)
        score = [accuracy, precision, recall, f1, sen, spe, auc_score]
        fpr, tpr, thresholds = roc_curve(Y_test, proba)
        plot_lines = [fpr, tpr, auc_score]

        return score,plot_lines

    def train(self,X,Y,clf,params):  
        
        posi=list(Y).count(1)
        nega=list(Y).count(0)
        total=posi+nega
        #加入class_weight
        weight_for_0 = (1 / nega)*(total)/2.0 
        weight_for_1 = (1 / posi)*(total)/2.0
        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto') 

        
        clf = load_model('origin.h5') # origin model
        

        history = clf.fit(X, Y,batch_size=params['batch_size'], epochs=params['nb_epoch'], class_weight = class_weight, validation_split=0.2, shuffle=False, callbacks=[reduce_lr]) 
        return clf, history


class ANN():
    def __init__(self):
        
        self.model =  keras.models.Sequential()
        self.initializer = initializers.glorot_uniform(seed=0) # set seed
        
    def compile_network(self, input_num, neuron_num, layer_num):
         
        input_ly = layers.Dense(neuron_num, input_dim = input_num, activation='relu',kernel_initializer=self.initializer)
        output_ly = layers.Dense(1, activation='sigmoid',kernel_initializer=self.initializer)
        
        for i in range(layer_num):
            if i==0: 
                self.model.add(input_ly)
#                 self.model.add(layers.Dropout(0.2))
            elif i<(layer_num-1):
                ly = layers.Dense(neuron_num, activation='relu',kernel_initializer=self.initializer)
                self.model.add(ly)
#                 self.model.add(layers.Dropout(0.2))
            else: 
                self.model.add(output_ly)
                
        
        optimizer = keras.optimizers.Adam(learning_rate=0.005)
        self.model.compile(loss = tf.losses.BinaryCrossentropy(), optimizer=optimizer,metrics=["acc"])
        
#         weights = self.model.get_weights()
#         weights[0] = weights[0]*np.array([0,1]*50)
#         weights[1] = weights[1]*np.array([0,1]*50)
#         self.model.set_weights(weights)
        
        self.model.save('origin.h5')
        return self.model
    

ML = machine_learning()

#Cross-validation
params = {'batch_size':500, 'nb_epoch':50}
ann = ANN().compile_network(input_num, neuron_num, layer_num)
scores, aucs, plot_data, mean_sen, mean_spe = ML.cv_train(X_train, Y_train, ann, params, 10, [])

#Model training
params = {'batch_size':500, 'nb_epoch':50}
ann = ANN().compile_network(input_num, neuron_num, layer_num)
CD_model, history = ML.train(X_train, Y_train, ann, params)

