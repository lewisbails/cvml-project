# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 18:51:14 2018

@author: Lewis Bails
"""
""" This script classifies the bottleneck vectors from Finetune using different
    models to fine which is best for the problem at hand """


#General dependencies
from itertools import product
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

#Deps for neural networks
from keras import callbacks
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.wrappers.scikit_learn import KerasClassifier

#Deps for SVM
from sklearn import svm

#Deps for XGBoost
from xgboost import XGBClassifier

#Deps for ELM
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RandomLayer

#Deps for LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#Deps for PNN
from neupy import environment
from neupy.algorithms import RBFKMeans,PNN

def load_labels():
    
    """ Load the labels into numpy arrays """ 
        
    #Read CSV into Pandas dataframe
    labels_train_df = pd.read_csv('C:/Users/Hew/Documents/Python/CVML_CNN/keras_model/Train/trainLbls.csv',header=None)
    labels_valid_df = pd.read_csv('C:/Users/Hew/Documents/Python/CVML_CNN/keras_model/Validation/valLbls.csv',header=None)

    #label arrays
    label_train = np.array(labels_train_df.values.flatten())
    label_valid = np.array(labels_valid_df.values.flatten())
    
    return label_train,label_valid

def get_cnn_vectors(path,mode=None):
    
    """ Loads CNN vectors given in original dataset, not from MobileNet """
    """ Possibly unused function if choosing to take MobileNet route """
    
    if mode=='Train':
        cnn_vectors_df = pd.read_csv(path+f'/{mode}'+r'/trainVectors.csv',header=None)
        labels_df = pd.read_csv(path+f'/{mode}'+r'/trainLbls.csv',header=None)
        
    elif mode=='Validation':
        cnn_vectors_df = pd.read_csv(path+f'/{mode}'+r'/valVectors.csv',header=None)
        labels_df = pd.read_csv(path+f'/{mode}'+r'/valLbls.csv',header=None)
        
    elif mode=='Test':
        cnn_vectors_df = pd.read_csv(path+f'/{mode}'+r'/testVectors.csv',header=None)  
   
    cnn_vectors_dict = {}
    cnn_vectors = np.array(cnn_vectors_df.transpose().values)
    
    if mode!='Test':
        labels = np.array(labels_df.values.flatten())

    
    if mode!='Test':
        for index in range(len(labels)):
            if labels[index] not in cnn_vectors_dict.keys():
                cnn_vectors_dict[labels[index]]=[cnn_vectors[index]]
            else:
                cnn_vectors_dict[labels[index]].append(cnn_vectors[index])
                
    else:
        cnn_vectors_dict['unknown']=[]
        for entry in cnn_vectors:
            cnn_vectors_dict['unknown'].append(entry)
            
    return cnn_vectors_dict


def get_arrays_from_dictionary(dictionary):
    
    """ Converts bottlenecks and labels into a dictionary """    
    
    vectors = []
    labels = []
    
    if 'unknown' in dictionary.keys():
        for entry in dictionary['unknown']:
            vectors.append(entry)
    
    else:
        for number in range(1,30):
            if number in dictionary.keys():
                for entry in dictionary[number]:
                    vectors.append(entry)
                    labels.append(number)
                
    vectors = np.array(vectors)
    labels = np.array(labels)
    
    return labels,vectors


class LossHistory(callbacks.Callback):

    ''' Batch History Function '''
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))


def model_param_selector(X_train,y_train,X_valid,y_valid,clf,param_grid,name='unknown_model'):
    
    """ Uses a grid search on the input parameter dictionary to find best
        parameters for the model passed in """
    
    X_final = np.concatenate((X_train,X_valid))
    y_final = np.concatenate((y_train,y_valid))
    
    splits = [(range(len(X_train)),range(len(X_train),len(X_final)))]

#    grid_result = GridSearchCV(clf,param_grid=param_grid,cv=splits,scoring='accuracy')
    grid_result = GridSearchCV(clf,param_grid=param_grid,cv=splits)
    
    grid_result.fit(X_final,y_final)

    print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
    np.save(f'{name}_results_.npy',grid_result.cv_results_)    

    return grid_result.best_params_,grid_result.best_estimator_


def create_model(hidden_layers=1,hidden=[],optimizer='adam',activation='relu',dropout=0.2):
    
    """ Used with the Keras wrapper function for grid searches on neural net
        hyperparameters. Simply initialises model. """
    
    model = models.Sequential()

    if hidden==[]:
        model.add(layers.Dense(30,activation='softmax',input_shape=(1024,)))
    elif len(hidden)==1:
        model.add(layers.Dense(hidden[0],activation=activation,input_shape=(1024,)))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(30,activation='softmax'))
    elif len(hidden)==2:
        model.add(layers.Dense(hidden[0],activation=activation,input_shape=(1024,)))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(hidden[1],activation=activation))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(30,activation='softmax'))   
            
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model
 
    
""" STEP 1: PRE-PROCESSING """

# 1 - Load

y_train,X_train = get_arrays_from_dictionary(np.load('mb_trainwdist_bottlenecks_finetuned_dict.npy').item())
y_valid,X_valid = get_arrays_from_dictionary(np.load('mb_valid_bottlenecks_finetuned_dict.npy').item())
X_test = np.load('mb_bottlenecks_test_finetuned.npy')

X_train = np.load('mb_bottlenecks_train_finetuned.npy')
y_train = load_labels()[0]


X_train_dict = np.load('mb_train_bottlenecks_finetuned_dict.npy').item()
# 3 - Normalise

scaler = StandardScaler()
scaler.fit(X_train)
train_mean = scaler.mean_
train_var = scaler.var_

X_train = X_train-train_mean
X_valid = X_valid-train_mean
X_test = X_test-train_mean

# 4 - Fix imbalance in classes using SMOTE

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)

# 6 - Convert labels to one-hot (if required in model)

y_train_ohl = to_categorical(y_train)
y_valid_ohl = to_categorical(y_valid)


##############################################################################
""" MODELS """


""" MODEL 1: SVM """

#Initialise model
svm_ = svm.SVC(verbose=True,random_state=42)

#Parameters to grid search
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 20],
              'kernel' : ['rbf','linear']}

#Perform grid search, return best parameters and estimator
svm_parameters,svm_estimator = model_param_selector(X_train, y_train, X_valid, y_valid,
                                                    svm_,param_grid,'svm')

#Save best model
joblib.dump(svm_estimator,'svm_mb_ft_.pkl')

#Predict and evaluate performance
svm_predictions = svm_estimator.predict(X_valid)

svm_accuracy = accuracy_score(np.array(y_valid),np.array(svm_predictions))

svm_report = classification_report(np.array(y_valid),np.array(svm_predictions))

svm_confusion = confusion_matrix(np.array(y_valid),np.array(svm_predictions))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.imshow(svm_confusion, interpolation='nearest', cmap=plt.cm.Blues)

#Predict on test bottlenecks
test_labels = svm_estimator.predict(X_test)
labels_test_df = pd.DataFrame(test_labels)
labels_test_df.to_csv('results_svm_20_rbf_ft_en.csv',header=['Label'])



""" MODEL 2: RANDOM FOREST """
#
#rf = RandomForestClassifier(n_estimators = 100, random_state = 42,
#                            max_depth=None, max_features='auto',
#                            min_samples_split=2,min_samples_leaf=1,
#                            bootstrap=True,criterion='gini')
#
#param_grid = {"n_estimators": [1000],
#              "max_depth": [None],
#              "max_features": ['auto'],
#              "min_samples_split": [2],
#              "min_samples_leaf": [1],
#              "bootstrap": [True],
#              "criterion": ["gini", "entropy"]}
#    
#rf_parameters,rf_estimator = model_param_selector(X_train, y_train, X_valid, y_valid,
#                                                  rf,param_grid,'rf')
#
#joblib.dump(rf_estimator,'rf_mb_distortions_smote.pkl')
# 
#rf_predictions = rf_estimator.predict(X_valid)
# 
#rf_accuracy = accuracy_score(np.array(y_valid),np.array(rf_predictions))
#
#rf_report = classification_report(np.array(y_valid),np.array(rf_predictions))
#
#rf_confusion = confusion_matrix(np.array(y_valid),np.array(rf_predictions))
#
#y_test = rf_estimator.predict(X_test)
#
#y_test_df = pd.DataFrame(y_test)
#y_test_df.to_csv('results_rf_whoknows.csv',header=['Label'])

""" MODEL 3: MLP NEURAL NETWORK """

#Initialise model
nn = KerasClassifier(build_fn=create_model)

#Parameters to grid search
hidden_units = [200,400]
hidden_layers = [0,1]
hidden = []
for ii in hidden_layers:
    hidden.extend(list(product(hidden_units,repeat=ii)))
for entry in range(len(hidden)):
    hidden[entry]=list(hidden[entry])

param_grid = {'batch_size':[32],
              'epochs':[5,10,20],
              'optimizer':['RMSprop', 'Adagrad', 'Adadelta', 'Adamax'],
              'activation':['relu','tanh'],
              'dropout':[0.1,0.2,0.3],
              'hidden':hidden}

#Perform grid search, return best parameters and estimator
nn_parameters,nn_estimator = model_param_selector(X_train, y_train_ohl, X_valid, y_valid_ohl,
                                                nn,param_grid,'nn')

#Save best model
joblib.dump(nn_estimator,'nn_.pkl')


#Predict and evaluate performance
nn_predictions = nn_estimator.predict(X_valid,batch_size=len(y_valid_ohl))

nn_accuracy = accuracy_score(np.array(y_valid),np.array(nn_predictions))

nn_report = classification_report(np.array(y_valid),np.array(nn_predictions))

nn_confusion = confusion_matrix(np.array(y_valid),np.array(nn_predictions))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.imshow(nn_confusion, interpolation='nearest', cmap=plt.cm.Blues)

#Predict on test bottlenecks
test_labels = nn_estimator.predict(X_test)
labels_test_df = pd.DataFrame(test_labels)
labels_test_df.to_csv('results_nn_ft_en.csv',header=['Label'])

""" MODEL 4: XGBoost """
#
#xgb = XGBClassifier(learning_rate=0.3, n_estimators=50, max_depth=5,
#                    min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                    objective= 'multi:softmax', scale_pos_weight=1,seed=27)
#
#param_grids = [{'max_depth':range(3,10,2),'min_child_weight':range(1,12,2)},
#               {'gamma':[i/10.0 for i in range(0,4)]},         
#               {'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]},
#               {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}]
#
#xgb_parameters = []
#
#i = 0
#
#for param_grid in param_grids:
#    xgb_variable = model_param_selector(X_train, y_train, X_valid, y_valid,
#                                        xgb,param_grid,f'xgb_selector_{i}')[0]
#    xgb_parameters.append(xgb_variable)
#    i+=1
#
#
## Now we can build our estimator from the best parameters    
#xgb_estimator = XGBClassifier(learning_rate=0.01, n_estimators=500, max_depth=5,
#                              min_child_weight=11, reg_alpha=0.1, gamma=0, subsample=0.7, colsample_bytree=0.6,
#                              objective= 'multi:softmax', scale_pos_weight=1,seed=27)
#
#xgb_estimator.fit(X_train,y_train)
#
#joblib.dump(xgb_estimator,'xgb_0.01_500_5_11_0.1_0_0.7_0.7.pkl')
#
#xgb_predictions = xgb_estimator.predict(X_valid)
#
#xgb_accuracy = accuracy_score(np.array(y_valid),np.array(xgb_predictions))
#
#xgb_report = classification_report(np.array(y_valid),np.array(xgb_predictions))
#
#xgb_confusion = confusion_matrix(np.array(y_valid),np.array(xgb_predictions))

""" MODEL 6: PNN """

#Initialise model
pnn = PNN(std=10, verbose=True)

#Parameters to grid search
X_range = np.amax(X_train)-np.amin(X_train)

std = []
for xx in [0.25,0.5,1,2]:
    std.append(X_range*xx)
param_grid = {'std':std}

#Perform grid search, return best parameters and estimator
pnn_parameters,pnn_estimator = model_param_selector(X_train, y_train, X_valid, y_valid,
                                                    pnn,param_grid,'pnn')

#Save best model
joblib.dump(pnn_estimator,'pnn_.pkl')

#Predict and evaluate performance
pnn_predictions = pnn_estimator.predict(X_valid)

pnn_accuracy = accuracy_score(np.array(y_valid),np.array(pnn_predictions))

pnn_report = classification_report(np.array(y_valid),np.array(pnn_predictions))

pnn_confusion = confusion_matrix(np.array(y_valid),np.array(pnn_predictions))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.imshow(pnn_confusion, interpolation='nearest', cmap=plt.cm.Blues)

#Predict on test bottlenecks
test_labels = pnn_estimator.predict(X_test)
labels_test_df = pd.DataFrame(test_labels)
labels_test_df.to_csv('results_pnn_ft_en.csv',header=['Label'])

""" MODEL 6: ELMC"""

#Parameters to grid search
X_range = np.amax(X_train)-np.amin(X_train)
rbf_width = []
for xx in [0.01,0.1,1,10]:
    rbf_width.append(X_range*xx)
    
n_hidden = [50,100,200,400,800]
alphas = [0.2,0.4,0.6,0.8,1]
rbf_widths = rbf_width
activation_funcs = ['tanh','sigmoid','gaussian']
rl = [[aa,bb,cc,dd] for aa in n_hidden for bb in alphas for cc in rbf_widths for dd in activation_funcs]
rls = []
for combo in rl:
    rls.append(RandomLayer(n_hidden=combo[0],alpha=combo[1],rbf_width=combo[2],activation_func=combo[3]))
    
param_grid = {'hidden_layer':rls}
    
#Instantiate model
elmc = GenELMClassifier(hidden_layer=rls[0])

#Perform grid search, return best parameters and estimator
elmc_parameters,elmc_estimator = model_param_selector(X_train, y_train, X_valid, y_valid,
                                                      elmc,param_grid,'elmc')

#Save best model
joblib.dump(pnn_estimator,'elmc_.pkl')

#Predict and evaluate performance
elmc_predictions = elmc_estimator.predict(X_valid)

elmc_accuracy = accuracy_score(np.array(y_valid),np.array(elmc_predictions))

elmc_report = classification_report(np.array(y_valid),np.array(elmc_predictions))

elmc_confusion = confusion_matrix(np.array(y_valid),np.array(elmc_predictions))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.imshow(elmc_confusion, interpolation='nearest', cmap=plt.cm.Blues)

#Predict on test bottlenecks
test_labels = elmc_estimator.predict(X_test)
labels_test_df = pd.DataFrame(test_labels)
labels_test_df.to_csv('results_elm_ft_en.csv',header=['Label'])


''' MODEL 7: LDA '''
#
##Instantiate model
#lda_estimator = LDA()
#
##Train the model
#lda_estimator.fit(X_train,y_train)
#
##Predict and evaluate the model
#lda_predictions = lda_estimator.predict(X_valid)
#
#lda_accuracy = accuracy_score(np.array(y_valid),np.array(lda_predictions))
#
#lda_report = classification_report(np.array(y_valid),np.array(lda_predictions))
#
#lda_confusion = confusion_matrix(np.array(y_valid),np.array(lda_predictions))
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.imshow(lda_confusion, interpolation='nearest', cmap=plt.cm.Blues)


""" END OF MAIN CODE """






















""" OUTSIDE TESTING OF MLP """

#Parameters


X_train_final = np.concatenate((X_train,X_valid))
y_train_ohl_final = np.concatenate((y_train_ohl,y_valid_ohl))

BATCH_SIZE=32
EPOCHS=20
HIDDEN_1=400
#HIDDEN_2=600
DROPOUT=0.2
DATA_TRAIN=X_train
OHL_TRAIN=y_train_ohl
DATA_VALID=X_valid
OHL_VALID=y_valid_ohl
DATA_TEST=X_test

#Initialise
model = models.Sequential()

#Input Layer
model.add(layers.Dense(HIDDEN_1,activation='tanh',input_shape=(1024,)))

#Hidden layers
model.add(layers.Dropout(DROPOUT))
#model.add(layers.LeakyReLU(alpha=0.3))
#model.add(layers.Dense(HIDDEN_2,activation='tanh'))
#model.add(layers.Dropout(DROPOUT))
#model.add(layers.LeakyReLU(alpha=0.3))

#Output layers
model.add(layers.Dense(len(OHL_TRAIN[0]),activation='softmax'))

#Summary
model.summary()

#Compile
model.compile(optimizer='Adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# STEP 12: TRAIN THE MODEL

history = LossHistory()
results_train = model.fit(DATA_TRAIN,OHL_TRAIN,validation_data=(DATA_VALID,OHL_VALID),
                          epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=[history],
                          shuffle=True)



#results_valid = model.predict(DATA_VALID)
#labels_valid = np.argmax(results_valid,axis=1)
#nn_report = classification_report(np.array(y_valid),np.array(labels_valid))
#labels_test_df = pd.DataFrame(labels_valid)
#labels_test_df.to_csv('results_nn_blerg.csv',header=['Label'])
#validation_data=(DATA_VALID,OHL_VALID)
#y_test = nn_estimator.predict(X_test)
#labels_test_df = pd.DataFrame(y_test)
#labels_test_df.to_csv('results_nn_tanh_32_0.3_20_400_adagrad.csv',header=['Label'])