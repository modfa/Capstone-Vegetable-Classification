#!/usr/bin/env python
# coding: utf-8

# ## Steps to Download the Dataset from Kaggle
# For more details, please read this -
# https://www.kaggle.com/general/74235

# In[1]:


get_ipython().system('pip install -q kaggle')


# In[2]:


get_ipython().system('mkdir ~/.kaggle')


# In[3]:


get_ipython().system('cp /content/kaggle.json ~/.kaggle/')


# In[4]:


get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[5]:


get_ipython().system('kaggle datasets list')


# In[6]:


get_ipython().system('kaggle datasets download misrakahmed/vegetable-image-dataset/')


# In[7]:


get_ipython().system('mkdir data')
get_ipython().system('unzip vegetable-image-dataset.zip -d data > /dev/null')
get_ipython().system('rm vegetable-image-dataset.zip')


# ## Importing the Necessary Packages

# In[8]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


from tensorflow.keras.preprocessing.image import load_img 


# ## EDA for Images Data

# In[10]:


path = '/content/data/Vegetable Images/train/Bitter_Gourd'
name = '0002.jpg'
fullname = path + '/' + name
load_img(fullname)


# In[11]:


load_img(fullname, target_size=(299, 299))


# 
# Pre-Trained Neural Network
# 
# Let's apply a pre-trained neural network with imagenet classes.
# 
# We'll use Xception, but any other architecture will work as well.
# 
# Check here for a list of available models:
# 
#     https://keras.io/api/applications/
#     https://www.tensorflow.org/api_docs/python/tf/keras/applications
# 
# We'll need to import 3 things:
# 
#     the model itself (Xception)
#     the preprocess_input function that takes an image and prepares it
#     the decode_predictions that converts the predictions of the model into human-readable classes
# 
# 

# In[12]:


from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions


# In[13]:


# Let's load the model. The pre-trained model expects 299x299 input
model = Xception(
    weights='imagenet',
    input_shape=(299, 299, 3)
)


# In[14]:


# 

# Next,

    # we load the image using the load_img function
    # convert it to a numpy array
    # make it a batch of one example


img = load_img(fullname, target_size=(299, 299))
x = np.array(img)
x.shape


# In[15]:


X = np.array([x])
X.shape


# In[16]:


X = preprocess_input(X)


# In[17]:


pred = model.predict(X)


# In[18]:


pred.shape


# In[19]:


pred[0, :10]


# In[20]:


decode_predictions(pred)


# 
# ### Transfer learning
# 
# - Instead of loading each image one-by-one, we can use a data generator. Keras will use it for loading the images and pre-processing them
# 

# In[21]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[22]:


image_size = (150, 150)
batch_size = 32


# In[23]:


train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory(
    "/content/data/Vegetable Images/train",
    seed=1,
    target_size=image_size,
    batch_size=batch_size,
)


# In[24]:


validation_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = validation_gen.flow_from_directory(
    "/content/data/Vegetable Images/validation",
    seed=1,
    target_size=image_size,
    batch_size=batch_size,
)


# 
# 
# For fine-tuning, we'll use Xception with small images (150x150)
# 

# In[25]:


base_model = Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False
)

base_model.trainable = False


# In[26]:


inputs = keras.Input(shape=(150, 150, 3))

base = base_model(inputs, training=False)
vector = keras.layers.GlobalAveragePooling2D()(base)
outputs = keras.layers.Dense(15)(vector)

model = keras.Model(inputs, outputs)


# In[27]:


learning_rate = 0.01

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


# In[28]:


history = model.fit(train_ds, epochs=7, validation_data=val_ds)


# In[29]:


# Code to plot the training accuraacy and validation accuracy
plt.figure(figsize=(6, 4))

epochs = history.epoch
val = history.history['val_accuracy']
train = history.history['accuracy']

plt.plot(epochs, val, color='black', linestyle='solid', label='validation')
plt.plot(epochs, train, color='black', linestyle='dashed', label='train')

plt.title('Xception v1, lr=0.01')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.xticks(np.arange(10))

plt.legend()


plt.savefig('xception_v1_0_01.svg')

plt.show()


# In[30]:


# To make it easier for us, let's make a function for defining our model:
def make_model(learning_rate):
    base_model = Xception(
        weights='imagenet',
        input_shape=(150, 150, 3),
        include_top=False
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(150, 150, 3))

    base = base_model(inputs, training=False)
    vector = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(15)(vector)

    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    return model


# In[31]:


scores = {}

for lr in [0.0001, 0.001, 0.01, 0.1]:
    print(lr)

    model = make_model(learning_rate=lr)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[lr] = history.history

    print()
    print()


# In[32]:


for lr, hist in scores.items():
    #plt.plot(hist['accuracy'], label=('train=%s' % lr))
    plt.plot(hist['val_accuracy'], label=('val=%s' % lr))

plt.xticks(np.arange(10))
plt.legend()


# In[33]:


learning_rate = 0.001


# In[ ]:





# ### Checkpointing
# 
#     - Saving the best model only
#     - Training a model with callbacks
# 

# In[34]:


model.save_weights('model_v1.h5', save_format='h5')


# In[35]:


chechpoint = keras.callbacks.ModelCheckpoint(
    'xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[36]:


learning_rate = 0.001

model = make_model(learning_rate=learning_rate)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[chechpoint]
)


# ### Adding more layers
# 
#  Adding one inner dense layer
#  Experimenting with different sizes of inner layer
# 

# In[39]:


def make_model(learning_rate=0.01, size_inner=100):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    
    outputs = keras.layers.Dense(15)(inner)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[40]:


learning_rate = 0.001

scores = {}

for size in [10, 100, 1000]:
    print(size)

    model = make_model(learning_rate=learning_rate, size_inner=size)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[size] = history.history

    print()
    print()


# In[42]:


for size, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=('val=%s' % size))

plt.xticks(np.arange(10))
plt.legend()


# In[ ]:





#  ### Regularization and dropout
# 
#   - Regularizing by freezing a part of the network
#   - Adding dropout to our model
#   - Experimenting with different values
# 

# In[45]:


def make_model(learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(15)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[46]:


learning_rate = 0.001
size = 100

scores = {}

for droprate in [0.0, 0.2, 0.5, 0.8]:
    print(droprate)

    model = make_model(
        learning_rate=learning_rate,
        size_inner=size,
        droprate=droprate
    )

    history = model.fit(train_ds, epochs=11, validation_data=val_ds)
    scores[droprate] = history.history

    print()
    print()


# In[48]:


for droprate, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=('val=%s' % droprate))

# plt.ylim(0.78, 0.86)
plt.legend()


# In[50]:


hist = scores[0.0]
plt.plot(hist['val_accuracy'], label=0.0)

hist = scores[0.5]
plt.plot(hist['val_accuracy'], label=0.5)

plt.legend()
#plt.plot(hist['accuracy'], label=('val=%s' % droprate))


# In[ ]:





# ### Data augmentation
# 
#   - Different data augmentations
#   - Training a model with augmentations
#   - How to select data augmentations?
# 

# In[51]:


train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
#     vertical_flip=True,
)

train_ds = train_gen.flow_from_directory(
    '/content/data/Vegetable Images/train',
    target_size=(150, 150),
    batch_size=32
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    '/content/data/Vegetable Images/train',
    target_size=(150, 150),
    batch_size=32,
    shuffle=False
)


# In[52]:


learning_rate = 0.001
size = 100
droprate = 0.0

model = make_model(
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=10, validation_data=val_ds)


# In[53]:


hist = history.history
plt.plot(hist['val_accuracy'], label='val')
plt.plot(hist['accuracy'], label='train')

plt.legend()


# In[ ]:





# ## Training a larger model
# 
#     - Train a 299x299 model
# 

# In[59]:


def make_model(input_size=150, learning_rate=0.01, size_inner=100,
               droprate=0.5):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(15)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[60]:


input_size = 299


# In[61]:


train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_ds = train_gen.flow_from_directory(
    '/content/data/Vegetable Images/train',
    target_size=(input_size, input_size),
    batch_size=32
)


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = train_gen.flow_from_directory(
    '/content/data/Vegetable Images/validation',
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False)


# In[62]:


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v4_1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[63]:


learning_rate = 0.001
size = 100
droprate = 0.0

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=12, validation_data=val_ds,
                   callbacks=[checkpoint])


# In[1]:





# ### Using the model
# 
# - Loading the model
# - Evaluating the model
# - Getting predictions
# 

# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[3]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications.xception import preprocess_input


# In[4]:


test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    '/content/data/Vegetable Images/test',
    target_size=(299, 299),
    batch_size=32,
    shuffle=False
)


# In[5]:


model = keras.models.load_model('/content/xception_v4_1_10_0.998.h5')


# In[6]:


model.evaluate(test_ds)


# In[7]:


path = '/content/data/Vegetable Images/test/Cabbage/0957.jpg'


# In[8]:


img = load_img(path, target_size=(299, 299))


# In[9]:


import numpy as np


# In[10]:


x = np.array(img)
X = np.array([x])
X.shape


# In[11]:


X = preprocess_input(X)


# In[12]:


pred = model.predict(X)


# In[18]:


classes_test = test_ds.class_indices
classes_test


# In[19]:


classes = [
    'Bean',
    'Bitter_Gourd',
 'Bottle_Gourd',
 'Brinjal',
 'Broccoli',
 'Cabbage',
 'Capsicum',
 'Carrot',
 'Cauliflower',
 'Cucumber',
 'Papaya',
 'Potato',
 'Pumpkin',
 'Radish',
 'Tomato']


# In[20]:


dict(zip(classes, pred[0]))


# In[ ]:




