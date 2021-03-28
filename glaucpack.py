########################################################################################
#                        Autor skryptu: Michał Tomaszewski                             #
########################################################################################

import sys
import matplotlib.pyplot
import numpy as np 
import seaborn as sns
import sklearn.metrics as sklm
import keras
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import backend as K


def run_tests(model, generator):
    
    predictions = model.predict_generator(generator, verbose = 1)
    
    #assert predictions.shape[-1] == 1, 'Only binary classification is supported rn'
    
    if predictions.shape[-1] == 1:
        predicted_classes = np.round(predictions).astype('byte')
    else:
        predicted_classes = np.argmax(predictions, axis = 1)
    
    # Matching predictions to targets
    correct = 0
    for prediction, real_label in zip(predicted_classes, generator.classes):

        print('Prediction = {} | Real label = {}'.format(prediction, real_label))

        if prediction == real_label:
            correct += 1

    # Accuracy
    test_accuracy = correct*100/len(predicted_classes)
    print('Test accuracy = {}%'.format(test_accuracy))
    
    # Confusion matrix
    test_cm = sklm.confusion_matrix(generator.classes, predicted_classes)
    TN, FP, FN, TP = sklm.confusion_matrix(generator.classes, predicted_classes).ravel()

    # Precision, recall , F1-score
    test_report = sklm.classification_report(generator.classes, predicted_classes)
    print('Classification report:')
    print(test_report)

    # ROC, AUC score
    FP_rate, TP_rate, thresholds = sklm.roc_curve(generator.classes, predictions)
    test_auc = sklm.auc(FP_rate, TP_rate)

    # Plots
    matplotlib.pyplot.figure(figsize = (20,10))
    matplotlib.pyplot.subplot(121)
    matplotlib.pyplot.plot([0, 1], [0, 1], 'k--')
    matplotlib.pyplot.plot(FP_rate, TP_rate, label='AUC = {:.3f})'.format(test_auc))
    matplotlib.pyplot.xlabel('False positive rate')
    matplotlib.pyplot.ylabel('True positive rate')
    matplotlib.pyplot.title('ROC curve')
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.subplot(122)
    sns.heatmap(test_cm, cmap = 'coolwarm', annot = True)
    matplotlib.pyplot.title('Confusion matrix')
    matplotlib.pyplot.xlabel('Predicted Labels')
    matplotlib.pyplot.ylabel('True Labels')
    matplotlib.pyplot.show()

# MODEL TRAINING PROCESS VISUALIZATIONS - start
# functions borrowed from: F. CHollet - Deep Learning with Python

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def visualize_training(history_dict, smoothed = False, smooth_factor = 0.8):
    
    acc = history_dict['acc']
    loss = history_dict['loss']
    val_acc = history_dict['val_acc']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    
    if smoothed:
        matplotlib.pyplot.figure(figsize = (12,12))
        matplotlib.pyplot.subplot(221)
        matplotlib.pyplot.plot(epochs, smooth_curve(loss, smooth_factor), 'bo', label = 'Train loss')
        matplotlib.pyplot.plot(epochs, smooth_curve(val_loss, smooth_factor), 'b', label = 'Val loss')
        matplotlib.pyplot.title('Val and train loss')
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.ylabel('Loss')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.subplot(222)
        matplotlib.pyplot.plot(epochs, smooth_curve(acc, smooth_factor), 'bo', label = 'Train accuracy')
        matplotlib.pyplot.plot(epochs, smooth_curve(val_acc, smooth_factor), 'b', label = 'Val accuracy')
        matplotlib.pyplot.title('Val and train accuracy')
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.ylabel('Accuracy')
        matplotlib.pyplot.legend()

        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.figure(figsize = (12,12))
        matplotlib.pyplot.subplot(221)
        matplotlib.pyplot.plot(epochs, loss, 'bo', label = 'Train loss')
        matplotlib.pyplot.plot(epochs, val_loss, 'b', label = 'Val loss')
        matplotlib.pyplot.title('Val and train loss')
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.ylabel('Loss')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.subplot(222)
        matplotlib.pyplot.plot(epochs, acc, 'bo', label = 'Train accuracy')
        matplotlib.pyplot.plot(epochs, val_acc, 'b', label = 'Val accuracy')
        matplotlib.pyplot.title('Val and train accuracy')
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.ylabel('Accuracy')
        matplotlib.pyplot.legend()

        matplotlib.pyplot.show()