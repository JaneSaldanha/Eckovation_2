#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from keras.layers import Dense,Flatten,Conv2D
from keras.layers import MaxPooling2D,Dropout
from keras.utils import np_utils,print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
import keras.backend as K


# In[ ]:


data=pd.read_csv("train_foo.csv")
dataset=np.array(data)
np.random.shuffle(dataset)
X=dataset
Y=dataset
X=X[:,1:2501]
Y=Y[:,0]


# In[ ]:


X_train=X[0:12000,:]
X_train=X_train/255.
X_test=X[12000:13201,:]
X_test=X_test/255.


# In[ ]:


Y=Y.reshpae(Y.shape[0],1)
Y_train=Y[0:12000,:]
Y_train=Y_train.T
Y_test=Y[12000:13201,:]
Y_test=Y_test.T


# In[ ]:


print("number of training examples="+str(X_train.shape[0]))
print("number of test examples="+str(X_test.shape[0]))
print("X_train shape"+str(X_train.shape))
print("Y_train shape"+str(Y_train.shape))
print("X_test shape:"+str(X_test.shape))
print("Y_test shape:"+str(Y_test.shape))


# In[ ]:


image_x=50
image_y=50
train_y=np_utils.to_categorical(Y_train)
test_y=np_utils.to_categorical(Y_test)
train_y=train_y.reshape(test_y.shape[1],test_y.shape[2])
test_y=test_y.reshape(test_y.shape[1],test_y.shape[2])
X_train=X_train.reshape(X_train.shape[0],50,50,1)
X_test=X_test.reshape(X_test.shape[0],50,50,1)
print("X train shape:"+str(X_train.shape))
print("X test shape:"+str(X_test.shape))


# In[3]:


def keras_model(image_x,image_y):
    num_of_claases=12
    model=Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(image_x,image_y,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Conv2D(64,(5,5),activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same'))
    model.add(Flatten())
    model.add(Dnese(1024,activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    filepath="face-rec256.h5"
    checkpoint1=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callbacks_list=[checkpoint1]
    return model,callbacks_list


# In[ ]:


model,callbacks_list=keras_model(image_x,image_y)
model.fit(X_train,train_y,validation_data=(X_test,test_y),epoch=10,batch_szie=64,callbacks=callbacks_list)
scores=model.evaluate(X_test,test_y,verbose=0)
print("CNN Error: %.2ff%%" % (100-scores[1])*100))
print_summary(model)
model.save('handEmo.h5')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




