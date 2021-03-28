########################################################################################
#                     Script author: Michał Tomaszewski                                #
########################################################################################

import numpy as np 
import seaborn as sns
import sklearn.metrics as sklm
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential, load_model, Model
from keras.applications import VGG16
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import preprocess_input as vgg16_preprocessing

from support_modules.glaucoma_functions import run_tests, smooth_curve, visualize_training


class MetricsPack(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        
        self.validation_data = val_generator
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []
        self.lowest_FN = np.sum(val_generator.classes)
        self.acceptable_FP = np.round(0.5*(len(val_generator.classes) - self.lowest_FN))

    def on_epoch_end(self, epoch, logs={}):
        
        val_generator = self.validation_data
        
        batch_size = val_batch_size
        breakpoint = len(val_generator) - 1
        
        
        if type(self.model.layers[-1]) is keras.engine.sequential.Sequential:
            class_layer_config = self.model.layers[-1].layers[-1].get_config()
        else:
            class_layer_config = self.model.layers[-1].get_config()

        # different handlers for different functions in the last node
        if class_layer_config['activation'] == 'sigmoid':
            
            val_data = np.zeros((len(val_generator)*batch_size, 256, 256, 3))
            val_labels = np.zeros((len(val_generator)*batch_size))
        
            for batch_id, (x, y) in enumerate(val_generator):
                val_data[batch_id*batch_size:(batch_id+1)*batch_size] = x
                val_labels[batch_id*batch_size:(batch_id+1)*batch_size] = y

                if batch_id == breakpoint:
                    break

            score = np.asarray(self.model.predict(val_data))
            predict = np.round(np.asarray(self.model.predict(val_data)))
            targ = val_labels

            go_further = True


        elif class_layer_config['activation'] == 'softmax' and class_layer_config['units'] == 2:
            print('got here')
            val_data = np.zeros((len(val_generator)*batch_size, 256, 256, 3))
            val_labels = np.zeros((len(val_generator)*batch_size, 2))
        
            for batch_id, (x, y) in enumerate(val_generator):
                val_data[batch_id*batch_size:(batch_id+1)*batch_size] = x
                val_labels[batch_id*batch_size:(batch_id+1)*batch_size, :] = y

                if batch_id == breakpoint:
                    break
            
            score = np.asarray(self.model.predict(val_data))
            predict = np.argmax(score, axis = 1)
            targ = np.argmax(val_labels, axis = 1)
            
            score = np.max(score, axis = 1)
            
            go_further = True
            
        else:
            print('Only binary classification is supported.')
            go_further = False


        # calculating metrics
        if go_further:
            epoch_auc = sklm.roc_auc_score(targ, score)
            epoch_confusion = sklm.confusion_matrix(targ, predict)
            epoch_precision = sklm.precision_score(targ, predict)
            epoch_recall = sklm.recall_score(targ, predict)
            epoch_f1 = sklm.f1_score(targ, predict)

            self.auc.append(epoch_auc)
            self.confusion.append(epoch_confusion)
            self.precision.append(epoch_precision)
            self.recall.append(epoch_recall)
            self.f1s.append(epoch_f1)

            epoch_FNs = epoch_confusion.ravel()[2]
            epoch_FPs = epoch_confusion.ravel()[1]

            # saving the model with lowest FN yet
            if epoch_FNs < self.lowest_FN and epoch_FPs <= self.acceptable_FP:
                filename = self.model.layers[0].name + '_low_FN_checkpoint.hdf5'
                self.model.save(filename)
                
                self.lowest_FN = epoch_FNs
                
                print('Found new low for FN. Saving this model to {}'.format(filename))
            else:
                pass

            print('Metrics on epoch end')
            print('AUC = {}'.format(epoch_auc))
            print('Precision = {}'.format(epoch_precision))
            print('Recall = {}'.format(epoch_recall))
            print('F1 = {}'.format(epoch_f1))
            print('False negativess = {}'.format(epoch_FNs))

            # If in JupyterNotebbok this would show a heatmap every interation
            # sns.heatmap(epoch_confusion, cmap = 'coolwarm', annot = True)
            # plt.xlabel('Predicted Labels')
            # plt.ylabel('True Labels')
            # plt.show()


train_dir = r''
val_dir = r''
test_dir = r''
 
# datagens
aggresive_augmentation = True
if aggresive_augmentation:
    train_datagen = ImageDataGenerator(#rescale = 1/255, # not needed with vgg16_preprocessing
                                       rotation_range = 40,
                                       width_shift_range = 0.1,
                                       height_shift_range = 0.1,
                                       zoom_range = 0.1,
                                       horizontal_flip = True,
                                       preprocessing_function = vgg16_preprocessing,
                                       fill_mode = 'reflect')  
else:
    train_datagen = ImageDataGenerator(#rescale = 1/255,
                                       horizontal_flip = True,
                                       vertical_flip = True,
                                       preprocessing_function = vgg16_preprocessing)

test_datagen = ImageDataGenerator(#rescale = 1/255,
                                  preprocessing_function = vgg16_preprocessing)


# MODEL DEFINITION
target_size = (256, 256)
input_shape = target_size + (3,)

transfer_learning = True
if transfer_learning:
    conv_base = VGG16(weights = 'imagenet',
                include_top = False,
                input_shape = input_shape)
    
    for layer in conv_base.layers:
        layer.trainable = False
    
else:
    conv_base = VGG16(weights = None,
                     include_top = False,
                     input_shape = input_shape)
    
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# prep generators
train_batch_size = 10
train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size = target_size,
                                                   batch_size = train_batch_size,
                                                   class_mode = 'binary', 
                                                   shuffle = True)

val_batch_size = 6
val_generator = test_datagen.flow_from_directory(val_dir,
                                                target_size = target_size,
                                                batch_size = val_batch_size,
                                                class_mode = 'binary')

test_generator = test_datagen.flow_from_directory(test_dir, target_size = target_size, 
                                                 batch_size = 1, shuffle = False, class_mode = 'binary')

metrics = MetricsPack()

model.compile(optimizer = Adam(),
             loss = 'binary_crossentropy',
             metrics = ['acc'])

val_loss_callback = ModelCheckpoint('train_demo_test_chckpt.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True)
early_stopper = EarlyStopping(monitor = 'val_loss', patience = 10, min_delta = 0.1)


train_bool = True
if train_bool:
    history = model.fit_generator(train_generator, validation_data = val_generator,
                        epochs = 1,
                        verbose = 1,
                        callbacks = [metrics, val_loss_callback])


# UNFREEZING SOME LAYERS
for layer in model.layers[0].layers[:15]:
    layer.trainable = False
for layer in model.layers[0].layers[15:]:
    layer.trainable = True
    
for layer_id, layer in enumerate(model.layers[0].layers):
    print(layer_id, layer.name, layer.trainable)

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics = ['acc'])

val_loss_callback = ModelCheckpoint('train_demo_test_chckpt.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True)

train_bool = True
if train_bool:
    history = model.fit_generator(train_generator, validation_data = val_generator,
                                epochs = 2,
                                verbose = 1,
                                callbacks = [metrics, val_loss_callback])


# whether to use the last trained model or use a saved model that was trained earilier
model_in_memory = True
if not model_in_memory: 
    model = load_model('equal_vgg16_tuning_checkpoint.hdf5')
    run_tests(model, test_generator)
else:
    run_tests(model, test_generator)

visualize_training(history.history, smoothed = True)