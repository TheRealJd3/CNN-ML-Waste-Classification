# Code by Jason Shawn D' Souza R00183051
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import itertools
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import GridSearchCV
import tensorflow.keras.layers
from tensorflow.keras.layers import Conv2D, Activation, Dropout, MaxPooling2D   
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.inception_v3 import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from keras import backend as K
from sklearn.datasets import load_files
from keras.applications import  inception_v3
import pickle
from pathlib import Path
import time

np.random.seed(183051)

# Trashnet Dataset used for training and general test
train_dir = Path('D:\\Programming\\CIT\\Labs\\Research_Project\\Implementation\\DATASET\\DATASET\\TRAIN')
test_dir = Path('D:\\Programming\\CIT\\Labs\\Research_Project\\Implementation\\DATASET\\DATASET\\TEST')

# RecycleNet Dataset
new_test_dir =Path('D:\\Downloads\\trashnet')

# Single Image
single_image_path = Path('D:\\Downloads\\test')

# Constants

NUM_EPOCHS = 100
ALEX_NET = 'AlexNet'
VGG = 'VGG'
INCEPTION = 'InceptionV3'
batchSize = 128


def load_dataset(path):
    """
    Loading datasets
    """
    # Getting whether folder is training or testing to print
    data_type = str(path).split("\\")[-1].lower()
    print(f"[INFO] Loading {data_type} dataset ...")
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    # storing and returning files, targets and labels
    # i.e file names, features of img
    print(f"[INFO] Done loading {data_type} dataset")
    return files, targets, target_labels


def convert_images_to_array(files):
    """
    Convert images to int arrays
    """
    width, height, channels = 100, 100, 3
    # define train and test data shape
    images_as_array = np.empty((files.shape[0], width, height, channels), dtype=np.uint8)
    for idx, file in enumerate(files):
        img = cv2.imread(file)
        # As images have different size, resizing all images to have same shape of image array
        res = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        images_as_array[idx] = res
        if len(files) == 1:
            img_gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            cv2.imshow("evaluation", img_gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return images_as_array


def model_train(x_train, y_train, x_valid, y_valid, model_type):
    """
    Train the model
    """
    print("Type of model : {}".format(model_type))
    wasteDataGenerator = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each input mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.2, # Randomly zoom image 
            rescale = 1.0/255, # Rescale/Normalize image
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    # using the imagedatagenerator defined above to
    # artificially generate more data(training and validation)
    train_generator = wasteDataGenerator.flow(x_train, y_train, batch_size=batchSize)
    valid_generator = wasteDataGenerator.flow(x_valid, y_valid, batch_size=batchSize)

    model = Sequential()
    #############
    #   VGG  ####
    #############                           
    if model_type == 'VGG':
        model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu',
                        input_shape=(100, 100, 3), name='conv1_1'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv1_2'))
        model.add(tensorflow.keras.layers.BatchNormalization(name='bn1'))
        model.add(MaxPooling2D(pool_size=(2, 2),name = 'maxpool0'))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',name = 'conv2_1'))
        model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', name = 'conv2_2'))
        model.add(tensorflow.keras.layers.BatchNormalization(name='bn2'))
        model.add(MaxPooling2D(pool_size=(2, 2),name = 'maxpool1'))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',name = 'conv3_1'))
        model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',name = 'conv3_2'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',name = 'conv3_3'))
        model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',name = 'conv3_4'))
        model.add(tensorflow.keras.layers.BatchNormalization(name='bn3'))
        model.add(MaxPooling2D(pool_size=(2, 2),name = 'maxpool2'))

        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',name = 'conv4_1'))
        model.add(Conv2D(256, kernel_size=(3, 3),activation='relu',name = 'conv4_2'))
        model.add(tensorflow.keras.layers.BatchNormalization(name='bn4'))
        model.add(MaxPooling2D(pool_size=(2, 2),name = 'maxpool3'))

        model.add(tensorflow.keras.layers.Flatten(name='fc_1'))
        model.add(tensorflow.keras.layers.Dense(512, activation='relu',name = 'Dense1'))

        model.add(tensorflow.keras.layers.Flatten(name='fc_2'))
        model.add(tensorflow.keras.layers.Dense(256, activation='relu',name = 'Dense2'))
        filepath = 'waste-model-VGGinspired.hdf5'
    elif model_type == 'AlexNet':
        #############
        #   AlexNet##  
        #############
        # First Layer
        model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4,4), kernel_initializer='he_normal',
                        input_shape=(100, 100, 3), activation='relu', name='conv0'))
        model.add(tensorflow.keras.layers.BatchNormalization(name='bn0'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same', name = 'maxpool0'))
        # Second layer
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization(name='bn1'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same', name = 'maxpool1'))
        
        # Third Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same',activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization(name='bn2'))

        
        # Fourth Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization(name='bn3'))
        
        # Fifth Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        model.add(tensorflow.keras.layers.BatchNormalization(name='bn4'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same', name = 'maxpool4'))
        
        # FC Layers
        # first
        model.add(tensorflow.keras.layers.Flatten(name='fc1'))
        model.add(tensorflow.keras.layers.Dense(512, activation='relu',name = 'Dense0'))
        # Second
        model.add(tensorflow.keras.layers.Flatten(name='fc2'))
        model.add(tensorflow.keras.layers.Dense(256, activation='relu',name = 'Dense1'))
        # Third
        model.add(tensorflow.keras.layers.Flatten(name='fc3'))
        model.add(tensorflow.keras.layers.Dense(128, activation='relu',name = 'Dense2'))
        filepath = 'waste-model-AlexNetinspired.hdf5'
    else:
        print('Not yet implemented')
    # 2 filters/Each class - organic and recyclable
    model.add(tensorflow.keras.layers.Dense(2, activation='softmax', name='classification'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    # val_loss  - value being monitored for improvement
    # min_delta - Abs value and is the min change required before we stop
    # patience  - Number of epochs we wait before stopping
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=15,
                              verbose=1,
                              restore_best_weights=True)
    ReduceLR = ReduceLROnPlateau(patience=3, verbose=1)
    callbacks = [earlystop, checkpoint, ReduceLR]
    # history = model.fit(x_train, y_train, epochs=200, callbacks=callbacks,validation_data=(x_valid, y_valid))
    model.summary()
    history = model.fit_generator(train_generator,epochs=NUM_EPOCHS, verbose=1, callbacks=callbacks, validation_data=valid_generator)
    pickle_out = open("Trained_cnn_history.pickle", "wb")
    pickle.dump(history.history, pickle_out)
    pickle_out.close()
    return history


def load_model_plot(history):
    """
    From train model generate plots
    """
    pickle_in = open("Trained_cnn_history.pickle", "rb")
    saved_history = pickle.load(pickle_in)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def model_evaluation(y_test, predicted_classes, title):
    """
    Plotting Confusion Matrix and printing Classification Report
    """
    # Confusion Matrix code inspiration from 
    # https://www.kaggle.com/raajparekh/cnn-from-scratch-inspired-from-vgg
    y_test = ['O' if x == 0 else 'R' for x in y_test]
    predicted_classes = ['O' if x == 0 else 'R' for x in predicted_classes]
    confusion_mtx = confusion_matrix(y_test, predicted_classes)
    print(confusion_mtx)
    plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{title} confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['O', 'R'], rotation=90)
    plt.yticks(tick_marks, ['O', 'R'])
    thresh = confusion_mtx.max() / 2.
    for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
        plt.text(j, i, confusion_mtx[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_mtx[i, j] > thresh else "red")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print("Classification Report :\n {}".format(classification_report(y_test, predicted_classes)))


def evaluate(x_test, y_test):
    """
    Predict classes needs float instead of uint8 so x_test needs to be float
    This is for CNN ONLY
    """
    model = load_model('waste-model-AlexNetinspired.hdf5')
    model.load_weights('waste-model-AlexNetinspired.hdf5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss :', score[0])
    print('Test Accuracy :', score[1])
    predicted_classes = model.predict_classes(x_test)
    model_evaluation(y_test, predicted_classes, "CNN")


def feature_extraction_eval(x_train, y_train, x_test, y_test, model_type):
    """
    Carrying out feature extraction and classification/evaluation process
    """
    if model_type == VGG:
        model = load_model('waste-model-VGGinspired.hdf5')
        model.load_weights('waste-model-VGGinspired.hdf5')
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('Dense1').output)
    elif model_type == ALEX_NET:
        model = load_model('waste-model-AlexNetinspired.hdf5')
        model.load_weights('waste-model-AlexNetinspired.hdf5')
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('Dense0').output)
    elif model_type == INCEPTION:
        initial_model = InceptionV3(weights='imagenet', include_top =False, input_shape =(100, 100,3))
        intermediate_layer_model = tf.keras.Model(inputs=initial_model.input, outputs=initial_model.get_layer('mixed1').output)
        # intermediate_layer_model = InceptionV3(weights='imagenet', include_top =False, input_shape =(100, 100,3))
    intermediate_layer_model.summary()
    train_feature_data = intermediate_layer_model.predict(x_train)
    train_feature_data = np.array(train_feature_data)
    test_feature_data = intermediate_layer_model.predict(x_test)
    test_feature_data = np.array(test_feature_data)
    if model_type == INCEPTION:
        train_feature_data = train_feature_data.reshape(train_feature_data.shape[0], -1)
        test_feature_data = test_feature_data.reshape(test_feature_data.shape[0], -1)

    # UN COMMENT TILL ABOVE

    # LOGISTIC
    print("[INFO] Begin Fitting Logistic Regression")
    logistic_reg = LogisticRegression(random_state=42).fit(train_feature_data, y_train)
    print("[INFO] End Fitting Logistic Regression")
    predictions = logistic_reg.predict(test_feature_data)
    if len(predictions) == 1:
        evaluate_single(predictions)
    else:
        print(f"LogisticRegression Score :{logistic_reg.score(test_feature_data, y_test)}")
        model_evaluation(y_test, predictions, "CNN("+model_type+")-LogisticRegression")
    #GNB
    print("[INFO] Begin Fitting GNB ")
    gnb = GaussianNB().fit(train_feature_data, y_train)
    print("[INFO] End Fitting GNB")
    predictions = gnb.predict(test_feature_data)
    if len(predictions) == 1:
        evaluate_single(predictions)
    else:
        print(f"Gaussian Naive Bayes Score :{gnb.score(test_feature_data, y_test)}")
        model_evaluation(y_test, predictions, "CNN("+model_type+")-GNB")

    # SVM
    best_svm_clf = SVC(C = 0.1, decision_function_shape='ovo', gamma='auto',kernel='linear')
    train_samples = x_train.shape[0]
    new_x_train = x_train.reshape(train_samples, -1)
    test_samples = x_test.shape[0]
    new_x_test = x_test.reshape(test_samples, -1)
    # print(f"Train Shape : {new_x_train.shape, y_train.shape}")
    print("[INFO] Begin fitting SVM Model")
    best_svm_clf.fit(train_feature_data, y_train)
    print("[INFO]  Done fitting SVM Model")
    predictions = best_svm_clf.predict(test_feature_data)
    if len(predictions) == 1:
        evaluate_single(predictions)
    else:
        print(f"SVM Score :{best_svm_clf.score(test_feature_data, y_test)}")
        model_evaluation(y_test, predictions, "CNN("+model_type+")-SVM")

    # KNN
    best_knn_clf = KNeighborsClassifier(metric='manhattan', n_neighbors=84, weights='distance')
    # {'metric': 'manhattan', 'n_neighbors': 84, 'weights': 'distance'} with a score of  0.6633979072528323
    print("[INFO] Begin fitting KNN Model")
    best_knn_clf.fit(train_feature_data, y_train)
    print("[INFO]  Done fitting KNN Model")
    predictions = best_knn_clf.predict(test_feature_data)
    if len(predictions) == 1:
        evaluate_single(predictions)
    else:
        print(f"KNN Score :{best_knn_clf.score(test_feature_data, y_test)}")
        model_evaluation(y_test, predictions, "CNN("+model_type+")-KNN")
    



def evaluate_single(prediction):
    """
    Evaluation for a single image
    """
    print(f"Note 0 is Organic 1 is Recyclable ")
    res = 'Organic'
    print(f"Predicted Class is {prediction[0]}")
    if int(prediction[0]) == 1:
        res = 'Recyclable'
    print(f"Predicted {res}")


def tf_init():
    """
    Fixes Tensorflow CUDNN errors
    """
    config = tf.compat.v1.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    # to log device placement (on which device the operation ran)
    config.log_device_placement = True
    # Set up new tf session
    sess = tf.compat.v1.Session(config=config)
    # set this TensorFlow session as the default session for Keras
    tf.compat.v1.keras.backend.set_session(sess)
    print("Initialized Tensorflow {}".format(tf.__version__))
    

def run(train, verbose_flag, mode):
    tf_init()
    x_train, y_train, target_labels = load_dataset(train_dir)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    x_train = np.array(convert_images_to_array(x_train))
    x_valid = np.array(convert_images_to_array(x_validate))
    x_train_float = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    single_file = np.array(['D:/Downloads/dataset-resized/O/O_01_SELF.jpeg'])
    # Set what model to use
    CNN_MODEL = ALEX_NET
    if verbose_flag:
        if train:
            print(f'Training set shape : {x_train.shape}')
            print(f'Training set size : ', x_train.shape[0])
            print("x_train shape: " + str(x_train.shape))
            print("x_train shape: " + str(y_train.shape))
            print("x_validate shape: " + str(x_validate.shape))
            print("y_validate shape: " + str(y_validate.shape))
            print('Validation set shape : ', x_valid.shape)
    if train:
        history = model_train(x_train, y_train, x_valid, y_validate,CNN_MODEL)
        load_model_plot(history)
    # FULL TEST
    if mode == 'techsash':
        x_test, y_test, _ = load_dataset(test_dir)
        x_test = np.array(convert_images_to_array(x_test))
        x_test_float = x_test.astype('float32')
        feature_extraction_eval(x_train_float, y_train, x_test_float, y_test, CNN_MODEL)
    elif mode == 'trashnet':
        new_x_test, new_y_test, new_target_labels = load_dataset(new_test_dir)
        new_x_test = np.array(convert_images_to_array(new_x_test))
        new_x_test_float = new_x_test.astype(np.float64)
        feature_extraction_eval(x_train_float, y_train, new_x_test_float, new_y_test, CNN_MODEL)
    elif mode == 'cnn':
        x_test, y_test, _ = load_dataset(test_dir)
        x_test = np.array(convert_images_to_array(x_test))
        x_test_float = x_test.astype('float32')
        evaluate(x_test_float, y_test)
    elif mode == 'single':
        # Single Image
        trial_x_single, trial_y_single, target_labels = load_dataset(single_image_path)
        trial_x_single = np.array(convert_images_to_array(trial_x_single))
        trial_x_single_float = trial_x_single.astype(np.float64)
        feature_extraction_eval(x_train_float, y_train, trial_x_single_float, trial_y_single, CNN_MODEL)
    else:
        print("[ERROR] Not Yet Implemented")


if __name__ == "__main__":
    run(False, False, 'techsash')
