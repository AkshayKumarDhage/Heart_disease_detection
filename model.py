#importing libraries
import numpy as np
import pandas as pd

#importing dataset
df = pd.read_csv('heart.csv')

#data preprocessing for categorical features
from sklearn.preprocessing import StandardScaler

dataset = df
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

stdSclr = StandardScaler()
col2scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[col2scale] = stdSclr.fit_transform(dataset[col2scale])


#Spliting dataset into train set and test set for prediction
from sklearn.model_selection import train_test_split
y = dataset['target']
X = dataset.drop(['target'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=0)

#training KNN for final prediction with k=8
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print("KNN accuracy score: {}" .format(accuracy_score(y_test, pred)*100))


#saving the model for future use
#from sklearn.externals import joblib
#joblib.dump(knn, 'knn_model.pkl')