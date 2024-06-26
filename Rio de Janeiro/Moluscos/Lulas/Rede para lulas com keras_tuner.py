import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split

#Dados para pesca indsutrial de lulas (39 anos com dados)
ano = np.array([1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022])
pesca_ind = np.array([20,129,18,115,20,12,87,23,57,30,6,21,286,488.5,252.5,345.5,637.5,214,387.5,389.5,444,421.5,507,572.5,2.167,28.509,22.627,210.691,201.510,76.178,94.943,25.068,0.023,70.803,44.520,38.862,13.997,25.392,13.432])


#Normalização dos dados
ano_media = np.mean(ano)
pi_media = np.mean(pesca_ind)

ano_std = np.sqrt( np.sum((ano - ano_media)**2)/38  )
pi_std = np.sqrt( np.sum((pesca_ind - pi_media)**2)/38  )

ano_normalizado = (ano - ano_media)/ano_std
pi_normalizado = (pesca_ind - pi_media)/pi_std



#Colocando dados normalizados no formato de entrada da rede
ano_normalizado = ano_normalizado.reshape((39,1))
pi_normalizado = pi_normalizado.reshape((39,1))

#Divisão entre teste e treino
ano_treino,ano_teste,pi_treino,pi_teste = train_test_split(ano_normalizado,pi_normalizado, test_size=0.15, shuffle=True)


#Hipermodelo
def build_model(hp):
    model = keras.Sequential()

    model.add(layers.Dense(1, activation=hp.Choice('activation',['relu','leaky_relu','tanh','sigmoid'])))      
    for i in range(hp.Int('num_layers',2,4,step=1)):
      model.add(layers.Dense(hp.Int(f'units_{i}',32,128,step=16), 
                             activation=hp.Choice('activation',['relu','leaky_relu','tanh','sigmoid'])))
    model.add(layers.Dense(1,activation='linear'))
    
    
    model.compile(optimizer=hp.Choice('optimizer',['adam','sgd','Nadam']),
                  loss='mse',
                  metrics=['accuracy'])

    return model


#Ajustando o hipermodelo
tuner = kt.BayesianOptimization(hypermodel=build_model, 
                                objective='loss',
                                max_trials=751,
                                overwrite=False,
                                project_name='Keras_tuner para lulas(sem dropout e batch norm)',
                                max_consecutive_failed_trials=20)

tuner.search(ano_treino,pi_treino, epochs=50, validation_data=(ano_teste,pi_teste))


#Salvando o melhor modelo e pesos
models = tuner.get_best_models(num_models=1)
best_model = models[0]
history = best_model.fit(ano_treino,pi_treino, epochs=500,verbose=1,validation_data=(ano_teste,pi_teste), batch_size=10)
best_model.save('/home/andre/Área de Trabalho/Projeto da pesca/modelo_projeto_pesca.keras')
best_model.save_weights('/home/andre/Área de Trabalho/Projeto da pesca/Lulas/pesos pesca ind/.weights.h5')
print(best_model.summary())
print(history.history.keys())


print(ano_teste*ano_std + ano_media)
print(best_model.predict(ano_teste)*pi_std + pi_media )

#Plot da função de perda
f1=plt.figure(1).add_subplot()
val_loss = history.history['val_loss']
loss = history.history['loss']
f1.plot(val_loss, label='val_loss')
f1.plot(loss, color='orange', label='loss')


#Funções de ativação
for i, layer in enumerate (best_model.layers):
    print (i, layer)
    try:
        print ("    ",layer.activation)
    except AttributeError:
        print('   no activation attribute')


#plot dos dados originais e predições
f2 = plt.figure(2).add_subplot()
f2.scatter(ano,pesca_ind)

y_pred = best_model.predict(ano_normalizado)*pi_std + pi_media
f2.plot(ano,y_pred)



plt.legend()
plt.show()













