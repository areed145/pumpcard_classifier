#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import glob
import random

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.regularizers import l2
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample


# In[2]:


input_files = glob.glob('arrays/*.npz')
print(len(input_files))


# In[3]:


Xs = []
ys = []

for file in input_files:
    loaded = np.load(file)
    Xs.append(loaded['X'])
    ys.append(loaded['y'])
    
X = np.array(Xs)
y = np.array(ys)


# In[4]:


samples = 200
Xs = []
ys = []
for class_name in np.unique(y):
    Xc = X[y==class_name]
    yc = y[y==class_name]
    for sample in range(samples):
        idx = random.randint(0,len(Xc)-1)
        Xs.append(Xc[idx])
        ys.append(yc[idx])


# In[5]:


X = np.array(Xs)
y = np.array(ys)
y = to_categorical(y)


# In[6]:


print(len(X))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)


# In[ ]:


shape = X.shape
print(X.shape)
print(X_train.shape)
print(X_test.shape)

categories = y.shape[1]
print(categories)


# In[ ]:


model = Sequential([
    Conv2D(
        64, (3, 3),
        activation='relu',
        padding='same',
        input_shape=(shape[1], shape[2], shape[3]),
    ),
    Conv2D(
        64, (3, 3), 
        strides=(2,2),
        activation='relu',
        padding='same',
    ),
    MaxPooling2D(
        pool_size=2
    ),
    Dropout(0.25),
    Conv2D(
        128, (3, 3),
        activation='relu',
        padding='same',
    ),
    Conv2D(
        128, (3, 3), 
        strides=(2,2),
        activation='relu',
        padding='same',
    ),
    MaxPooling2D(
        pool_size=2
    ),
    Dropout(0.25),
    Flatten(),
    Dense(
        512, 
        activation='relu'
    ),
    Dropout(0.5),
    Dense(
        categories, 
        activation='softmax'
    ),
])


# In[ ]:


model.compile(
    optimizer=SGD(lr=0.01),
#     optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)


# In[ ]:


model.summary()


# In[ ]:

checkpoint = ModelCheckpoint(
	'model.h5', 
	verbose=1, 
	monitor='accuracy',
	save_best_only=True, 
	mode='auto'
) 

model.fit(
    X_train, y_train,
#     batch_size=100,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)


# In[ ]:


model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
print('saved model to disk')

