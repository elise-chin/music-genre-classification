import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Preparation du dataset
data = pd.read_csv('data.csv')
data = data.drop(['filename'],axis=1)#On supprime la premiere colonne

#On change les noms des genres par des int
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
data.iloc[:, -1] = y

#On normalise le dataset
scaler = StandardScaler()
data = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

#On separe le dataset en train set et test set (80%/20%)
data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.2)