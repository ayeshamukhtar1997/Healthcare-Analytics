import pandas as pd

#Getting dataset

data = pd.read_csv('cleveland.csv', header = None)

data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

#Preprocessing dataset

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###########################################################################################################################################
print("Support Vector Machine")
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
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
print("Logistic Regression")

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
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
print("XGBoostClassifier")

from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = xg.predict(X_train)

for i in range(0, len(y_pred_train)):
    if y_pred_train[i]>= 0.5:       # setting threshold to .5
       y_pred_train[i]=1
    else:  
       y_pred_train[i]=0
       
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
print("CNN Classifier")
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

def create_model():
    
    model = Sequential([
                    Dense(16, input_dim=13,kernel_initializer='normal', activation = 'relu' ),
                    Dense(32,kernel_initializer='normal', activation='relu'),
                    Dense(2, activation='softmax') ])
    # compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()
print(model.summary())
history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=100, batch_size=10, verbose = 10)

from sklearn.metrics import classification_report, accuracy_score
categorical_pred = np.argmax(model.predict(X_test), axis=1)

print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))



"""
Model Accuracy and Loss History
%matplotlib inline
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

"""
