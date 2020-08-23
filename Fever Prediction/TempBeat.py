import pandas as pd
data = pd.read_csv('Fever.csv', names=['Temperature', 'Gender', 'Heartrate','Condition'])

#Plotting dataset
import matplotlib.pyplot as plt
# create a figure and axis
fig, ax = plt.subplots()

# scatter the sepal_length against the sepal_width
ax.scatter(data['Temperature'], data['Heartrate'])
# set a title and labels
ax.set_title('Temp Dataset')
ax.set_xlabel('Temperature')
ax.set_ylabel('Heartrate')


data = pd.read_csv('Fever1.csv', header = None)
data = data.apply(pd.to_numeric)
data.columns = ['Temperature', 'Gender', 'Heartrate','Condition']
### 0 = Normal, 1 = Fever, 2 = High Fever, 3 = Hypothermia
data.isnull().sum()
data['Condition'] = data.Condition.map({0: 'Normal', 1: 'Fever', 2:'High Fever', 3:'Hypothermia'})

#Preprocessing

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


###############################################################################################################################################
print("GuassianNB")

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

#confusion matrix of training and testing
print(cm_train)
print(cm_test)

#Training accuracy
print("Training Accuracy")
print((cm_train[0][0] + cm_train[1][1])/len(y_train))
print("Testing Accuracy")
print((cm_test[0][0] + cm_test[1][1])/len(y_test))

###############################################################################################################################################
print("Random Forest Classifier")


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)
#confusion matrix of training and testing
print(cm_train)
print(cm_test)

#Training accuracy
print("Training Accuracy")
print((cm_train[0][0] + cm_train[1][1])/len(y_train))
print("Testing Accuracy")
print((cm_test[0][0] + cm_test[1][1])/len(y_test))


###############################################################################################################################################
print("Decision Tree Classifier")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)


#confusion matrix of training and testing
print(cm_train)
print(cm_test)

#Training accuracy
print("Training Accuracy")
print((cm_train[0][0] + cm_train[1][1])/len(y_train))
print("Testing Accuracy")
print((cm_test[0][0] + cm_test[1][1])/len(y_test))




###############################################################################################################################################

import sys
import numpy as np
import sklearn
import matplotlib
import keras


from sklearn import model_selection
#Check for empty instances
data = data[~data.isin(['?'])]

X = np.array(data.drop(['target'], 1))
y = np.array(data['target'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

print("NN Classifier (Model 1)")####################################################################################

model = Sequential()
model.add(Dense(16, input_dim=3, activation = 'relu' ))
model.add(Dense(8,activation='relu'))               
model.add(Dense(3, activation='softmax'))
# compile model
adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print(model.summary())
model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=100, batch_size=10, verbose = 10)




print("NN Classifier (Model 2)")####################################################################################
model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=150, batch_size=5)
