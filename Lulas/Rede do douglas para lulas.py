import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import autokeras as ak
import keras
from keras import layers
from sklearn.model_selection import train_test_split

#Dados para pesca total(ind + art) de lulas
ano = np.array([1962,1963,1964,1966,1967,1968,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2017,2018,2019,2020,2021,2022])
total_pescado = np.array([22,57,71,21,70,602,126,304,353,488,147,152,296,199,171,228,118,167,322.5,592.5,265,473,651.5,398.5,476,476,545,528,629.5,645,19.727,63.039,33.307,231.284,210.388,85.774,304.782,27.585,127.763,78.423,80.826,43.419,82.212,76.866])

#Normalização dos dados
ano_media = np.mean(ano)
tp_media = np.mean(total_pescado)

ano_std = np.sqrt( np.sum((ano - ano_media)**2)/43  )
tp_std = np.sqrt( np.sum((total_pescado - tp_media)**2)/43  )

ano_normalizado = (ano - ano_media)/ano_std
tp_normalizado = (total_pescado - tp_media)/tp_std


#Colocando dados normalizados no formato de entrada da rede
ano_normalizado = ano_normalizado.reshape((44,1))
tp_normalizado = tp_normalizado.reshape((44,1))

#Divisão entre teste e treino
ano_treino,ano_teste,tp_treino,tp_teste = train_test_split(ano_normalizado,tp_normalizado, test_size=0.2, shuffle=True)

#Rede
model = tf.keras.Sequential([layers.Dense(256,activation='leaky_relu',input_shape=(1,)),
                             layers.Dense(128,activation='leaky_relu'),
                             layers.Dense(64,activation='leaky_relu'),
                             layers.Dense(32,activation='leaky_relu'),
                             layers.Dense(1,activation='leaky_relu')])
opt = keras.optimizers.Nadam(learning_rate=0.01)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

history=model.fit(ano_treino,tp_treino,validation_data=(ano_teste,tp_teste),epochs=100,verbose=1)


#Predição na amostra de teste
tp_pred = model.predict(ano_teste)*tp_std + tp_media

print(ano_teste*ano_std + ano_media)
print(tp_pred)

#Predição nos dados ausentes
print('Pescado em 1965: ',model.predict(np.array([[(1965 - ano_media)/ano_std]]))*tp_std + tp_media)
print('Pescado em 1995: ',model.predict(np.array([[(1995 - ano_media)/ano_std]]))*tp_std + tp_media)
print('Pescado em 2016: ',model.predict(np.array([[(2016 - ano_media)/ano_std]]))*tp_std + tp_media)

#Plot da função de perda e acurácia
val_loss = history.history['val_loss']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
plt.plot(val_loss, label='val_loss')
plt.plot(loss, color='orange', label='loss')
plt.plot(val_acc, color='purple', label='val_accuracy')

plt.legend()
plt.show()