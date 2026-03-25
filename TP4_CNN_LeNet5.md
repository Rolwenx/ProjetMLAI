#### TP4 : Convolutional Neural Networks (CNN)

#### ST2AIM - "AI & Machine Learning" - 2024/2025

#### Author : Youssef Ait El Mahjoub

<br/>

--------------------------------------------------------------------
## <font color="red"> Objectives of the Practical Activity  </font>
--------------------------------------------------------------------

- Retrieve, Split and standardize data
- Implement CNN architecture with tensorflow
- Fixe "tensorFlow" training optimizer to 'SGD' then to 'ADAM'
- Developp a LeNet5 architecture with different blocks
<br/>

-------------------------------------------------
## <font color="red"> Reminder questions  </font>
-------------------------------------------------

1. What is the difference between DNN and CNN ?
2. What is the classical architecture of a CNN ?
3. What is the purpose of convolution and pooling layers ?
4. How to regulate results to avoid Overfitting ?


-----------------------------------------------------------------
## <font color="red">  DataSets </font>
-----------------------------------------------------------------

- "sklearn.load_digits() returns an object of type Bunch, which is similar to a dictionary, but you can access its values via attributes. The key attributes of the Bunch object are:
    - 'data': a numpy array with the shape (nSamples = 1797, nFeatures = 64 = 8x8 pixels). Each row corresponds to a flattened digit image (originally 8x8 pixels, flattened into a 64-element vector).
    - 'target': a numpy array containing the corresponding labels, shaped (1797,), where each element is an integer representing the handwritten digit.
    - 'images': an array shaped (1797, 8, 8), where each element is an 8x8 matrix representing a digit image.
    - 'target_names': an array of all possible digits (0 to 9).
    - 'DESCR': a complete description of the dataset.
<br/>
<br/>

- "MNIST" - Mixed National Institute of Standards and Technology is a database of handwritten digits.
    - You need to download the database, or check for "mnist.npz" folder.
    - The dataset is of size (nSamples = 60 000, nFeatures = 784 = 28x28 pixels)
    - details in "https://www.tensorflow.org/datasets/catalog/mnist?hl=fr" 
<br/>
<br/>

-----------------------------------------------------------------
## <font color="red">  Image Classification - LeNet5 architecture </font>
-----------------------------------------------------------------

- You need to complete the "cnn_students.py" file.
- We will implement the LeNet5 Architecture : 
    - An input layer
    - one Convolution layer : 6 filters of size (5,5)
    - one MaxPooling Layer  : pool of size (2,2)
    - one Convolution layer : 16 filters of size (5,5)
    - one MaxPooling Layer  : pool of size (2,2)
    - Flatten Layer
    - Dense Layer 1         : 120 neurons
    - Dense Layer 2         : 84 neurons
    - Output Layer          : 10 neuros

1. Analyze the structure of the code to understand what is expected.
2. Make sure you already have : "load_digits()" and "mnist()" datasets
    - In "main" function, you need to have a section for the "load_digits()" (i.e DATASET1) and "mnist" (i.e DATASET2) 
    - Normalize correctly each dataset (as seen in previous practical activities)
    - Convert the nominal 'y' values to binary, by completing "to_categorical()" function
<br/>
<br/>

3. Implementing LeNet5 Architecture : 
    - Complete the function "Keras_CNN_LeNet5(cnn, X_train, y_train, X_test, y_test, opt="SGD")", that respects LetNet5 archtitecture. Example : 
        - Conv2D(filters=6, kernel_size=(5,5), activation='ReLu', kernel_initializer='he_uniform', padding="same")
        - MaxPooling2D((2,2), strides=2)
    - Fixe a bSize variable (i.e. batch size) to 32 for DATASET1 and 64 for DATASET2.
    - Train the models as "kerasCNN.fit(X_train, y_train, epochs=dnn.n_iterations, batch_size=bSize)"
    - Fixe the optimizer compiler to "tf.keras.optimizers.SGD(dnn.learning_rate)" or "tf.keras.optimizers.Adam(dnn.learning_rate)"
    - Comple the "main" function an run the model
<br/>
<br/>

4. Testing the model : 
    - For DATASET1 : 
        - Fix parameters: lr = 0.01, maxIteration = 10.
        - Verify that Optimizer="SGD" acheives approximatively <font color="red">70.33%</font> accuracy.
        - Verify that Optimizer="Adam" acheives approximatively  <font color="red">98.33%</font> accuracy.

    - For DATASET2 : 
        - Fix parameters: lr = 0.01, maxIteration = 10.
        - Verify that Optimizer="SGD" acheives approximatively <font color="red">97.50%</font> accuracy.
        - Verify that Optimizer="Adam" acheives approximatively  <font color="red">98.43%</font> accuracy.
<br/>
<br/>

5. Graphical representation of model performance during iterations :
    - Adapt the function "Keras_CNN_LeNet5()" to store history of loss function and accuracy. Use the instruction "history = kerasCNN.fit(X_train, y_train, ... , validation_data=(X_test, y_test) )".
    - Complete the function "plot_history(history)" wich displays two figures, one for 'Cross entropy values' and a second one for 'Accuracy' for each iteration. Each figure displays a plot for Training and a plot for Testing.
    - Call "plot_history(history)" as a last instruction.
    - Examine results, observe how training and testing accuracy evolves. 
    