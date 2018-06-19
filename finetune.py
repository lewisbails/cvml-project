# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:48:03 2018

@author: Lewis Bails
"""

"""""""""" Finetune """""""""""
"""
This script is used for finetuning MobileNet and enriching the original dataset
The process is broken down into steps after the function declarations.
"""

#Dependencies
import os
import pandas as pd
import numpy as np
from shutil import copy2
from keras import callbacks,optimizers,Model,models,layers
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt

def load_labels():
    
    """ Load the labels into numpy arrays """    
    
    #Read CSV into Pandas dataframe
    labels_train_df = pd.read_csv('C:/Users/Hew/Documents/Python/CVML_CNN/keras_model/Train/trainLbls.csv',header=None)
    labels_valid_df = pd.read_csv('C:/Users/Hew/Documents/Python/CVML_CNN/keras_model/Validation/valLbls.csv',header=None)

    #label arrays
    label_train = np.array(labels_train_df.values.flatten())
    label_valid = np.array(labels_valid_df.values.flatten())
    
    return label_train,label_valid

def get_bottlenecks(model,path,num_images,batch_size):

    """ Computes output from model given the image path and returns them in
        a numpy array """
    
    #Initialise image numbers
    images_left = range(1,num_images+1)
    
    for batch in range(num_images//batch_size +1):
        print(f'\nBatch {batch+1}/{num_images//batch_size +1}')
        #Compute batch image arrays and update remaining image numbers
        array_batch,images_left = bottleneck_batch(batch_size,path,images_left)
        print('\nGenerating...')
        if batch==0:
            #Compute bottlenecks from image arrays and add them to final array
            bottlenecks = np.squeeze(model.predict(array_batch,verbose=1))
            print('\nBottlenecks array initialised')
        else:
            bottlenecks = np.concatenate((bottlenecks,np.squeeze(model.predict(array_batch,verbose=1))))
            print('\nBatch added.')
        print('\n\n\nBatch bottlenecks complete!')
        
    return bottlenecks

def get_dictionary_from_arrays(vectors,labels,dic={}):
    
    """ Converts bottlenecks and labels into a dictionary """
    
    for index in range(len(labels)):
        if labels[index] not in dic.keys():
            dic[labels[index]]=[vectors[index]]
        else:
            dic[labels[index]].append(vectors[index])
            
    return dic
                
def get_arrays_from_dictionary(dictionary):
    
    """ Converts dictionary of labels and bottlenecks into separate vectors """
    
    vectors = []
    labels = []
    
    if 'unknown' in dictionary.keys():
        for entry in dictionary['unknown']:
            vectors.append(entry)
    
    else:
        for number in range(1,30):
            #add original data to list
            if number in dictionary.keys():
                for entry in dictionary[number]:
                    vectors.append(entry)
                    labels.append(number)
                
    vectors = np.array(vectors)
    labels = np.array(labels)
    
    return labels,vectors

def bottleneck_batch(batch_size,path,images_left):
    
    """ Converts batch of images to arrays and preprocesses them for MobileNet
    """
    
    if len(images_left)>=batch_size:
        numbers = images_left[:batch_size]
        images_left = images_left[batch_size:]
    else:
        numbers = images_left
        images_left = []
    
    print(f'\nCreating bottlenecks for Image{numbers[0]} to Image{numbers[-1]}')
    
    X_train = []
    for number in numbers:
        image_ = image.load_img(path+f'/Image{number}.jpg',target_size=(224,224))
        image_array = image.img_to_array(image_)
        X_train.append(preprocess_input(image_array))
        print(f'\n\tImage{number} preprocessed')
    X_train = np.array(X_train)

    return X_train,images_left
    
def train_batch(train_datagen,y_train,batch_size,train_numbers):
    
    """ Generate a batch of transformed image arrays for training the model """
    
    if len(train_numbers)>=batch_size:
        numbers = train_numbers[:batch_size]
        train_numbers = train_numbers[batch_size:]
    else:
        numbers = train_numbers
        train_numbers = []
    
    X_train_transformed = []
    for number in numbers:
        image_ = image.load_img(f'{os.getcwd()}'+f'/Train/TrainImages/Image{number}.jpg',target_size=(224,224))
        image_array = image.img_to_array(image_)
        transformed_array = train_datagen.random_transform(image_array)
        image.array_to_img(transformed_array).save(f'{os.getcwd()}'+f'\Train\TrainImages\mod\Image{number}_mod.jpg')
        X_train_transformed.append(preprocess_input(transformed_array))
    
    index = np.array(numbers)-1
    
    return np.array(X_train_transformed),y_train[index],train_numbers

def valid_batch(valid_datagen,y_valid,batch_size):
        
    """ Generates a batch of image arrays for model's validation """
    
    numbers = np.random.permutation(range(1,2299))[:batch_size]
    
    X_valid_transformed = []
    for number in numbers:
        image_ = image.load_img(f'{os.getcwd()}'+f'/Validation/ValidImages/Image{number}.jpg',target_size=(224,224))
        image_array = image.img_to_array(image_)
        transformed_array = train_datagen.random_transform(image_array)
        X_valid_transformed.append(preprocess_input(transformed_array))
    
    index = np.array(numbers)-1
    
    return np.array(X_valid_transformed),y_valid[index]

def test_batch(num_images):
    
    """ Generate a batch of image arrays for testing the model """
    
    X_test = []
    for number in num_images:
        image_ = image.load_img(f'{os.getcwd()}'+f'/Test/TestImages/Image{number}.jpg',target_size=(224,224))
        image_array = image.img_to_array(image_)
        X_test.append(preprocess_input(image_array))
    
    return np.array(X_test)
    


""" STEP 1: INSTANTIATE THE DATA GENERATORS """

train_datagen = ImageDataGenerator(rotation_range=180,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

""" STEP 2: LOAD LABELS FROM FILE """

y_train,y_valid = load_labels()

""" STEP 3: INSTANTIATE MOBILENET MODEL AND AUGMENT FOR BOTTLENECKS """

base_model = MobileNet(input_shape=(224,224,3),include_top=False)

for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.trainable)
    
x = base_model.output
bottleneck = GlobalAveragePooling2D()(x)

mobile_bottleneck = Model(inputs=base_model.input, outputs=bottleneck)

""" STEP 4: RETRIEVE BOTTLENECKS FOR TRAINING FC LAYER """

#Get bottlenecks from images
X_train = get_bottlenecks(mobile_bottleneck,f'{os.getcwd()}'+'\Train\TrainImages',5830,100)
np.save('mb_bottlenecks_train.npy',X_train)

X_valid = get_bottlenecks(mobile_bottleneck,f'{os.getcwd()}'+'\Validation\ValidImages',2298,100)
np.save('mb_bottlenecks_valid.npy',X_valid)

X_test = get_bottlenecks(mobile_bottleneck,f'{os.getcwd()}'+'\Test\TestImages',3460,100)
np.save('mb_bottlenecks_test.npy',X_test)

#Get bottleneck dictionaries from file
train_dict = get_dictionary_from_arrays(np.load('mb_bottlenecks_train.npy'),y_train,dic={})
np.save('mb_bottlenecks_train_dict.npy',train_dict)

valid_dict = get_dictionary_from_arrays(np.load('mb_bottlenecks_valid.npy'),y_valid,dic={})
np.save('mb_bottlenecks_valid_dict.npy',valid_dict)

#Get bottlenecks and arrays from file
train_dict = np.load('mb_train_bottlenecks_dict.npy').item()
y_train,X_train = get_arrays_from_dictionary(np.load('mb_train_bottlenecks_dict.npy').item())
y_valid,X_valid = get_arrays_from_dictionary(np.load('mb_valid_bottlenecks_dict.npy').item())


""" STEP 5: AUGMENT LABELS FOR OUTPUT ONE-HOT """

y_valid = to_categorical(y_valid)
y_train = to_categorical(y_train)

""" STEP 6: TRAIN FC """

head = models.Sequential()

head.add(layers.Dense(500,activation='linear',input_shape=(1024,)))
head.add(layers.Dropout(0.2))
head.add(layers.LeakyReLU(alpha=0.3))
head.add(Dense(30,activation='softmax'))

head.compile(optimizer='adamax',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

head.fit(X_train,y_train,
         epochs=2,
         batch_size=32,
         validation_data=(X_valid,y_valid),
         shuffle=True)

head.save_weights('top_model_mb.h5')

""" STEP 7: FIND THE MISCLASSIFIED INSTANCES AND RETRIEVE CLASS-REPORT """

#Generate report
y_valid_preds = np.argmax(head.predict(X_valid),axis=1)
y_valid_actual = np.argmax(y_valid,axis=1)
wrong = np.array(np.where(y_valid_preds!=y_valid_actual))[0]+1

head_report = classification_report(y_valid_actual,y_valid_preds)

#Create folder showing misclassified images
path = os.getcwd()+f'\Validation\ValidImages\wrong'
if not os.path.exists(path):
    os.makedirs(path)
for number in wrong:
    old_path = os.getcwd()+f'\Validation\ValidImages\Image{number}.jpg'
    copy2(old_path,path)

""" STEP 8: ATTACH FC TO AUGMENTED INCEPTION MODEL """

top_model = models.Sequential()

top_model.add(layers.Dense(500,activation='linear',input_shape=(1024,)))
top_model.add(layers.Dropout(0.2))
top_model.add(layers.LeakyReLU(alpha=0.3))
top_model.add(Dense(30,activation='softmax'))

top_model.load_weights('top_model_mb.h5')

""" STEP 9: INSTANTIATE NEW MOBILENET MODEL AND LOAD FINE-TUNED WEIGHTS """

model = Model(inputs=mobile_bottleneck.input,outputs=top_model(mobile_bottleneck.output))
model.load_weights('finetuned_mb_67.h5')

""" STEP 10: SET TRAINABLE LAYERS FOR FINE-TUNING """

for layer in model.layers[:69]:
    layer.trainable = False
for layer in model.layers[69:]:
    layer.trainable = True

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)
    
""" STEP 11: TRAIN NEW MOBILENET MODEL """

#Compile
model.compile(optimizer=optimizers.SGD(lr=0.001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Set training parameters
batch_size = 64
val_batch_size = 32
class_size = 500
val_acc = []
train_acc = []
epochs = 6

#Balance and expand training set
expansion_factors = {}
for key in train_dict.keys():
    expansion_factors[key]=(class_size//len(train_dict[key]))

image_numbers = [1]

for key in expansion_factors.keys():
    for i in range(expansion_factors[key]):
        indices = np.where(np.argmax(y_train[:5830],axis=1)==key)[0]
        new_image_numbers = np.array(indices)+1
        image_numbers = np.concatenate((image_numbers,new_image_numbers))

image_numbers = image_numbers[1:]

#Begin training
for epoch in range(1,epochs+1):
    
    print(f'\nEpoch {epoch}/{epochs}')

    train_numbers = np.random.permutation(image_numbers)
    train_len = len(train_numbers)
    
    print(f'\n{train_len} image numbers in the training set.')
    
    for batch in range(train_len//batch_size + 1):
    
        X_t,y_t,train_numbers = train_batch(train_datagen,y_train,batch_size,train_numbers)
        batch_acc = model.train_on_batch(X_t,y_t)
        train_acc.append(batch_acc)

        print(f'\n\tBatch {batch+1}/{train_len//batch_size +1}:\tacc:{batch_acc}')
        
#        if batch%10==0:
#            X_v,y_v = valid_batch(valid_datagen,y_valid,val_batch_size)
#            batch_val_acc = model.test_on_batch(X_v,y_v)
#            val_acc.append(batch_val_acc)
#            print(f'\t\t\tval_acc:{batch_val_acc}') 
    
    model.save_weights(f'finetuned_mb_71.h5')

""" STEP 12: GET NEW BOTTLENECKS OF FINETUNED MODEL """

#Define fine-tuned model
mobile_bottleneck_finetuned = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d_1').output)

#Get bottlenecks as before
X_train = get_bottlenecks(mobile_bottleneck_finetuned,f'{os.getcwd()}'+'\Train\TrainImages',5830,100)
np.save('mb_bottlenecks_train_finetuned.npy',X_train)

X_valid = get_bottlenecks(mobile_bottleneck_finetuned,f'{os.getcwd()}'+'\Validation\ValidImages',2298,100)
np.save('mb_bottlenecks_valid_finetuned.npy',X_valid)

X_test = get_bottlenecks(mobile_bottleneck_finetuned,f'{os.getcwd()}'+'\Test\TestImages',3460,100)
np.save('mb_bottlenecks_test_finetuned.npy',X_test)

#Get dictionaries
X_train_ft_dict = get_dictionary_from_arrays(X_train,np.argmax(y_train,axis=1),{})
np.save('mb_train_bottlenecks_finetuned_dict.npy',X_train_ft_dict)

X_valid_ft_dict = get_dictionary_from_arrays(X_valid,np.argmax(y_valid,axis=1),{})
np.save('mb_valid_bottlenecks_finetuned_dict.npy',X_valid_ft_dict)

""" STEP 13: GET BOTTLENECKS FROM DISTORTED IMAGES """

class_size = 900
batch_size = 32
dist_dict = {}

expansion_factors = {}
for key in train_dict.keys():
    expansion_factors[key]=(class_size//len(train_dict[key]))
    
train_numbers = [1]

for key in expansion_factors.keys():
    for i in range(expansion_factors[key]):
        indices = np.where(np.argmax(y_train[:5830],axis=1)==key)[0]
        new_train_numbers = np.array(indices)+1
        train_numbers = np.concatenate((train_numbers,new_train_numbers))

train_numbers = train_numbers[1:]

train_len = len(train_numbers)
            
for batch in range(train_len//batch_size + 1):
    print(f'\nBatch {batch+1}/{train_len//batch_size +1}')
    X_t,y_t,train_numbers = train_batch(train_datagen,y_train,batch_size,train_numbers)
    print('\nDistortions computed')
    bottles = mobile_bottleneck_finetuned.predict_on_batch(X_t)
    print('\nBottlenecks computed')
    dist_dict = get_dictionary_from_arrays(bottles,np.argmax(y_t,axis=1),dist_dict)
    print('\nDictionary updated')
        
np.save('mb_dist_bottlenecks_finetuned_dict.npy',dist_dict)

#dist_dict = np.load('mb_dist_bottlenecks_finetuned_dict.npy').item()

""" STEP 14: ADD THEM TO THE EXISTING TRAINING DATA BOTTLENECKS """

for number in range(1,30):
    #add distorted data to list
    if number in dist_dict.keys():
        X_train = np.concatenate((X_train,np.array(dist_dict[number])))
        y_train = np.concatenate((y_train,to_categorical([number]*len(dist_dict[number]),num_classes=30)))

new_train = {}
new_train = get_dictionary_from_arrays(X_train,np.argmax(y_train,axis=1),new_train)
np.save('mb_trainwdist_bottlenecks_finetuned_dict.npy',new_train)





""" The new bottlenecks are to be used in the 'multiple_models.py' code """
""" END OF MAIN CODE """