import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')
x = data.iloc[:,3:-1].values
y = data.iloc[:,-1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])

ohe = OneHotEncoder(categorical_features=[1])
x=ohe.fit_transform(x).toarray()
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
#initialising the ANN
classifier = Sequential()
#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6,
                     init = 'uniform',
                     activation = 'relu',
                     input_dim = 11))
#second hidden layer
classifier.add(Dense(output_dim = 6,
                     init = 'uniform',
                     activation = 'relu'))
#output layer
classifier.add(Dense(output_dim = 1,
                     init = 'uniform',
                     activation = 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting the dtaaset
classifier.fit(x_train,y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


