# HappyorSad
Deep CNN Image classifier using Tensorflow

# Building a data pipeline

In order to properly process data, we need to first set rules and recognize if any of the data types might interfere with cv2 reading the files.

Taking advantage of TensorFlow/Keras library, we can create batches of data without manually adding labels for the title, columns, and rows.

![Screenshot 2024-01-16 at 10 18 26](https://github.com/juhwani/HappyorSad/assets/7718853/75f14af3-f435-4489-a313-4ff4791083c4)

# Preprocessing Data

In order to optimize our models, we can actually divide the pixel dimensions by 255, which is the format we have set for the images. 

Then I split the data into training, evaluation, and test.

This is to make sure to track model's accuracy and allocating the appropriate amount of datasets to find the most fit.

# Building the Model

We have different layers added to the model: Conv2D, Maxpooling2D, Flatten, and Dense.
![Screenshot 2024-01-16 at 10 19 25](https://github.com/juhwani/HappyorSad/assets/7718853/98184c83-25c3-4dfa-910f-71cc73ca1186)

These layers formulate the neural network to train our model

# Training the Model

We train by 'fitting' the model using the batches data we have provided.

![Screenshot 2024-01-16 at 10 20 47](https://github.com/juhwani/HappyorSad/assets/7718853/ec662fc4-8e6e-47a3-8430-dcc10a8ff39f)

Ephoch shows how many times we iterate through the data.



# Evaluating 

We can visualize our losses and accuracy over the epochs and track how our model fits

![Screenshot 2024-01-16 at 10 21 25](https://github.com/juhwani/HappyorSad/assets/7718853/3ceaf158-82d0-4360-a055-910831219042)
![Screenshot 2024-01-16 at 10 21 40](https://github.com/juhwani/HappyorSad/assets/7718853/0662c1d1-8f2e-4eed-850c-ab0e512e3322)



