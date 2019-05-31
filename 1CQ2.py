#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.preprocessing import image
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from IPython.display import SVG
from keras.layers import Dense, Conv2D, MaxPooling2D,Dropout,BatchNormalization,Flatten


# In[ ]:


parapath = 'E:/OneDrive/2019_Semester_1/COMP3330/1CQ2/cell-images-for-detecting-malaria/cell_images/Parasitized2'
uninpath = 'E:/OneDrive/2019_Semester_1/COMP3330/1CQ2/cell-images-for-detecting-malaria/cell_images/Uninfected2'
parastized = os.listdir(parapath)
uninfected = os.listdir(uninpath)


# In[ ]:


data = []
label = []

for para in parastized:
    try:
        img =  image.load_img(parapath+para,target_size=(128,128))
        x = image.img_to_array(img)
        data.append(x)
        label.append(1)
    except:
        print("Can't add "+para+" in the dataset")
        
for unin in uninfected:
    try:
        img =  image.load_img(uninpath+unin,target_size=(128,128))
        data.append(x)
        label.append(0)
    except:
        print("Can't add "+unin+" in the dataset")  


# In[ ]:


data = np.array(data)
label = np.array(label)
print(sys.getsizeof(data))
print(data.shape)


# In[ ]:


data = data/255
x_train, x_test, y_train, y_test = train_test_split(data,label,test_size = 0.1,random_state=0)


# In[ ]:


def MalariaModel():
    model = Sequential()
    model.add(Conv2D(filters = 4, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a11', input_shape = (128, 128, 3)))  
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a12'))
    model.add(BatchNormalization(name = 'a13'))
    #input = (128,128,4)
    model.add(Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a21'))   
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a22'))
    model.add(BatchNormalization(name = 'a23'))
    #input = (64,64,8)
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a31'))   
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a32'))
    model.add(BatchNormalization(name = 'a33'))
    #input = (32,32,16)
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', name = 'fc1'))
    model.add(Dense(1, activation = 'sigmoid', name = 'prediction'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


modelone = MalariaModel()
modelone.summary()


# In[ ]:


output = modelone.fit(x_train, y_train,epochs=1, batch_size=1000)


# In[ ]:


preds = modelone.evaluate(x = x_test,y = y_test)
print("Test Accuracy : %.2f%%" % (preds[1]*100))


# In[ ]:


modelone.save('malariaCNNModel.h5')


# In[ ]:


modelpic = plot_model(modelone, to_file='model.png')
SVG(model_to_dot(modelone).create(prog='dot', format='svg'))


# In[ ]:





# In[ ]:





# In[ ]:




