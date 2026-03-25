import numpy as np
import tensorflow as tf
import math, pickle, sys
import matplotlib.pyplot as plt
from sklearn import datasets as datasetsSk
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, BatchNormalization
import time
from keras.datasets import cifar10
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#------------ Parameters -----------#
DATASET       = 3  # Set to CIFAR-10 by default

learningRate  = 0.001
maxIterations = 50

nHidden1      = 64      # Number of filters in first convolutional layer
nHidden2      = 128      # Number of filters in second convolutional layer
ConvKernel    = 3       # Size of filters in convolution layer
Poolkernel    = 2       # Size of filters in pooling layer

#---------- Helpers Functions  -------------#
def normalize(X, axis=-1, order=2):
    ''' Normalize the dataset X
    - Each vector ligne x (an entry of X) is normalized as x = (x / ||x||_2 )
    '''
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def plot_history(history):
    '''Plot the training and validation loss and accuracy'''
    # Extract loss and accuracy for training and validation sets
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1, len(loss) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def accuracy(y_true, y_pred):
    ''' Calculate the accuracy of a model with predicted values and actual values'''
    return (y_true == y_pred).mean()


def plot_image(images, labels, predictions, plot_all=True, num_images=20):
    '''Displays images in 'test' dataset, their labels, and predicted values'''
    name = "CIFAR10"
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    count = 0
    for index in range(len(images)):
        if count >= num_images:
            break

        image = images[index]
        label = np.argmax(labels[index])
        prediction = predictions[index]

        if plot_all or label == prediction:
            title = f'CNN {name}: Label: {classes[label]}, Predicted: {classes[prediction]}'
            file_name = f"./{name}/CNN_{name}_{classes[label]}_{index}.pdf"

            plt.figure(figsize=(5, 5))
            plt.imshow(image)
            plt.title(title)
            plt.grid(False)
            plt.axis('off')
            plt.savefig(file_name)
            plt.close()
            count += 1

# --------- Cross Entropy Error Class -------------#
class CrossEntropy:
    def __init__(self): pass

    def loss(self, y, p):
        '''Cross-Entropy Loss function for multiclass predictions'''
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.sum(y * np.log(p))

    def acc(self, y, p):
        ''' Accuracy between One-hot encoding : target value 'y' and predicted 'p' '''
        return accuracy(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        '''Gradient of Cross-Entropy function'''
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return p - y

# --------- ReLU activation Class -------------#
class ReLU():
    def __call__(self, x):
        '''ReLU activation function'''
        return np.maximum(0, x)

    def gradient(self, x):
        '''Derivative of the ReLU function'''
        return 1. * (x > 0)

# --------- Softmax activation Class --------------#
class Softmax():
    def __call__(self, x):
        '''Softmax function'''
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

# ---------------------- CNN parameters -------------------------------#
class ConvolutionNeuralNetwork():
    def __init__(self):
        '''Initialization of CNN "hyper-parameters" '''
        self.n_hidden1 = nHidden1
        self.n_hidden2 = nHidden2
        self.n_iterations = maxIterations
        self.learning_rate = learningRate
        self.hidden_activation = ReLU()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()
        self.Ckernel = ConvKernel
        self.Pkernel = Poolkernel

# ---------------------- CNN_LeNet5: with Keras in TensorFlow ------------------#
def Keras_CNN_LeNet5(cnn, X_train, y_train, X_test, y_test, optimizer):
    ''' Using TensorFlow library
    1- Create LeNet5 CNN model with tf.keras
    2- Fix algorithm optimizer (SGD, Adam) and error function
    3- Train the model
    4- Test the model
    5- Plot graphics
    '''

    h_activation = type(cnn.hidden_activation).__name__.lower()

    # Data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # 1- Creating CNN Model with Normalization and Dropout
    model = Sequential()
    model.add(Conv2D(cnn.n_hidden1, kernel_size=cnn.Ckernel, activation=h_activation, input_shape=(32, 32, 3),
                     padding='same'))
    model.add(BatchNormalization())  # Add normalization layer
    model.add(MaxPooling2D(pool_size=cnn.Pkernel))
    model.add(Dropout(0.3))  # Add dropout to reduce overfitting

    model.add(Conv2D(cnn.n_hidden2, kernel_size=cnn.Ckernel, activation=h_activation, padding='same'))
    model.add(BatchNormalization())  # Add normalization layer
    model.add(MaxPooling2D(pool_size=cnn.Pkernel))
    model.add(Dropout(0.3))  # Add dropout to reduce overfitting

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Output layer for 10 classes

    # 2- Optimizer and Learning Rate Scheduler
    if optimizer == "SGD":
        print("Optimizer: SGD")
        opt = tf.keras.optimizers.SGD(cnn.learning_rate, momentum=0.9)
    else:
        print("Optimizer: Adam")
        opt = tf.keras.optimizers.Adam(learning_rate=cnn.learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    #lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-5)


    # Train the model with augmented data
    start_time = time.time()
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),  # Training with augmented data
        epochs=cnn.n_iterations,
        validation_data=(X_test, y_test),
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

    # 4- Testing model
    test_loss, test_acc = model.evaluate(X_test, y_test)

    # 5- Get predictions
    predicted_classes = np.argmax(model.predict(X_test), axis=-1)

    # 6- Call plot_image function
    plot_image(X_test, y_test, predicted_classes)
    plot_history(history)
    return test_acc

if __name__ == "__main__":
    ###################  1- Importing DataSet ###############
    if DATASET == 3:
        #Data : cifar10.load_data()
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # Normalize the data
        X_train = X_train / 255.0
        X_test  = X_test / 255.0

        # Convert the labels to categorical vectors
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        print("After categorical\n", y_train)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        print("After categorical\n", y_test)

        print("Training      : ", X_train.shape)
        print("Test          : ", X_test.shape)

    ##################### 2- Creating Model #################
    cnn = ConvolutionNeuralNetwork()
    print(cnn.learning_rate)

    #################### 3- CNN with TensorFlow ##############
    accuracy = Keras_CNN_LeNet5(cnn, X_train, y_train, X_test, y_test, optimizer="Adam")

    print("Test accuracy is: ", accuracy)
