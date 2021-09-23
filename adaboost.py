import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

df = pd.read_csv('IRIS.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import AdaBoostClassifier
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
#y_pred = model.predict(X_test)


pickle.dump(model, open('iri.pkl', 'wb'))

