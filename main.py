# %% Building a data pipleline
import tensorflow as tf
import keras
import os 
import cv2
import imghdr

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus: # Makes it so I'm not using all of my gpu power to run this script
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'data' # Dir name for data
image_exts = ['jpeg','jpg','bmp','png'] # Acceptable iamge files

os.listdir(data_dir)

for image_class in os.listdir(data_dir): # Two image classes: Happy and sad
    if image_class == '.DS_Store': # Ignoring 
        continue
    for image in os.listdir(os.path.join(data_dir, image_class)): # Going through all the images in both classes
        image_path = os.path.join(data_dir, image_class, image) # Stores image path to a variable
        try:
            img = cv2.imread(image_path) # Loads image from the specified file. Determines by the content not by the file extension. If cannot be read then method returns an empty matrix
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print("Image not in ext list {}".format(image_path)) # If the type of image is not included in the acceptable image files then it is deleted.
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

import numpy as np

from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory('data') # generates*** an image dataset fro mthe given directory through a pipeline without manually adding columns and rowxs yourself 

data_iterator = data.as_numpy_iterator() # Access the generated content
#%%
batch = data_iterator.next() # Actually accessing the data. Basically images loaded into numpy arrays

# %%
batch[0]
# %%
batch[0].shape
# %%
batch[1] # in this case this is the labels. 1 or 0 represents happy or sad
# %%
fig, ax = plt.subplots(ncols=4, figsize = (20,20)) # Displays the batch images and their classes
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
# %% Preprocessing Data 
# For some recap our batches are divided into two parts: Images and their classes
batch[0]
batch[1]

# %% We need to be able to optimize our models which means we can divide our values by 255 (max)

#Scaling Data

data = data.map(lambda x,y: (x/255,y)) # x is image y is class
# %% Split Data: 

train_size = int(len(data)*0.7) # Used for training our model
val_size = int(len(data)*0.2) + 1 # Evaluate our model while training. Fine tune 
test_size = int(len(data) * 0.1) + 1 # Post Training 

# Above this is how much we are allocating to our processes



# %%

# Also make sure to shuffle the data before doing this
train = data.take(train_size) # How much data we are going to take during the partition
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# %% Building the model itself Actually doing deep learning

# import dependcies

from tensorflow.keras.models import Sequential # One input one output
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# Conv2D: 2D Convultional layer spatial convolution over images
# MaxPooling2D: Condensing layer: 
# 

# %%
model = Sequential()

# %%
model.add(Conv2D(16, (3,3),1, activation ='relu', input_shape = (256,256,3)))# Adding Convultion layer with input. 16 filters (3,3) pixels moving 1 pixel at a time. relu activation: Converting any negative numbers into zero and conserving postiive.    
model.add(MaxPooling2D()) # Take maximum values and gets the output

model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3),1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation = 'sigmoid')) # Takes anything to put it into from 0 to 1
# %%
model.compile('adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])

# adam is the optimizer and loss binary classification with accuracy metric

# %%
model.summary() # Shows our model with different layers and number of parameters
# %%
# Train

logdir = 'log'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir) # Save our model at a checkpoint call backs do that

# Actually training the data now.
# we're 'fitting' the data into model. takes train data so 'train'. epoch means how long. Validation data means we can see how well it is doing in real time. Log out all the information 
hist = model.fit(train, epochs = 20, validation_data = val, callbacks = [tensorboard_callback])

# %%
# Visualize our performance

fig = plt.figure()

plt.plot(hist.history['loss'], color = 'teal', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'orange', label = 'val_loss')
fig.suptitle('Loss', fontsize = 20)
plt.legend(loc = 'upper left')
plt.show()
# If loss goes up then it is over fitting 
# %%

fig = plt.figure()

plt.plot(hist.history['accuracy'], color = 'teal', label = 'accuracy')
plt.plot(hist.history['val_accuracy'], color = 'orange', label = 'val_accuracy')
fig.suptitle('Accuracy', fontsize = 20)
plt.legend(loc = 'upper left')
plt.show()

# %% Evaluation

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# Different measure classficiation problems



# %%
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y= batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
    

# %%
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
# %%
img = cv2.imread('happytest2.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
# %%
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
# %%
# Encapsulate into another set of array this is because we did them in batches before
np.expand_dims(resize, 0)
yhat = model.predict(np.expand_dims(resize/255,0))
# %%
print(yhat)

if yhat > 0.5:
    print('Predicted class is sad')
else:
    print('Predicted class is Happy')
# %%
