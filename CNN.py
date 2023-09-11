#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import datasets, layers, models 
from tensorflow.keras.datasets import fashion_mnist


# In[2]:


#preparing Dataset 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[3]:


#Sampling Validation Dataset 
train_index = np.arange(0, len(train_labels)-12000)
valid_index = np.arange(len(train_labels)-12000, len(train_labels))

valid_images = train_images[list(valid_index)]
valid_labels = train_labels[list(valid_index)]

train_images = train_images[list(train_index)]
train_labels = train_labels[list(train_index)]

print('Validation Image Dataset shape: ', valid_images.shape)
print('Validation Image Dataset label length: ', len(valid_labels))
print('Training Image Dataset shape: ', train_images.shape)
print('Training Image Dataset label length:', len(train_labels))


# In[4]:


#generating CNN 
model = models.Sequential()
model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(56, (3, 3), activation='relu'))


# In[5]:


#generating Flaten, Dense 
model.add(layers.Flatten())
model.add(layers.Dense(56, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))


# In[6]:


model.summary()


# In[7]:


model.compile(optimizer = "Adam",
             loss = "sparse_categorical_crossentropy",
             metrics = ['accuracy'])


# In[8]:


#model training 
model.fit(train_images, train_labels, 
          epochs = 10,
         batch_size = 32,
         validation_data = (valid_images, valid_labels))


# In[9]:


#model verificiation
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest Loss: ', test_loss)
print('\nTest Accuracy: ', test_acc)


# In[10]:


#model prediciton 
test_prediction = model.predict(test_images)


# In[14]:


#showing examples of test set for each class where the model misclassifies. 
misclassified_index = [] 
for i in range(len(test_labels)): 
    if np.argmax(test_prediction[i]) != test_labels[i]:
        misclassified_index.append(i)


# In[19]:


for class_label in range(10):
    for index in misclassified_index:
        if test_labels[index] == class_label:
            plt.figure()
            plt.imshow(test_images[index], cmap='gray')
            plt.title(f'Predicted: {class_names[np.argmax(test_prediction[index])]}, Acutal: {class_names[test_labels[index]]}')
            plt.show()
            break


# In[ ]:




