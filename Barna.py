import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


#Importamos datos
house_df = pd.read_csv(r'C:\Users\Alcalde\Downloads\idealista.csv') 

#Visualización
sns.scatterplot(x='m^2', y='Precios', data=house_df)

#Correlation 
f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(house_df.corr(), annot=True)
plt.show()
#Limpieza de datos
selected_features = ['m^2','Habitaciones']
X = house_df[selected_features]
y = house_df['Precios']

#Utilizamos scaler, lo escalamos de 0 a 1 para observar mejor los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

#Normalizando output
y = y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)

#Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25)

#Definiendo modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu',input_shape=(2, )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.summary()

model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2)

#Evaluando Modelo --> Tenemos como llaves el loss y val_loss
epochs_hist.history.keys()

#Gráfico
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso del Modelo durante el Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss']) #Ploteamos el entrenamiento de nuestro modelo

#Predicción
#Primero definimos una casa con sus respectivas entradas
#Hacemos una prueba
X_test_1 = np.array([[85,3]])

scaler_1 = MinMaxScaler()
X_test_scaled_1 = scaler_1.fit_transform(X_test_1)

#Haciendo predicción
y_predict_1 = model.predict(X_test_scaled_1)

#Revirtiendo Escalado para apreciar el precio correctamente escalado
y_predict_1 = scaler.inverse_transform(y_predict_1)