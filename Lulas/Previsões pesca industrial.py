import numpy as np
import matplotlib.pyplot as plt
import keras

#Dados disponíveis para pesca industrial
ano = np.array([1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022])
pesca_ind = np.array([20,129,18,115,20,12,87,23,57,30,6,21,286,488.5,252.5,345.5,637.5,214,387.5,389.5,444,421.5,507,572.5,2.167,28.509,22.627,210.691,201.510,76.178,94.943,25.068,0.023,70.803,44.520,38.862,13.997,25.392,13.432])

ano_media = np.mean(ano)
ano_std = np.sqrt( np.sum((ano - ano_media)**2)/38  )
ano_normalizado = (ano - ano_media)/ano_std

pi_media = np.mean(pesca_ind)
pi_std = np.sqrt( np.sum((pesca_ind - pi_media)**2)/38  )

#Dados a seram preditos e normalização
anos_sem_pesca_ind = np.array([1990,1991,1992,1993,1994,1995])

aspi_media = np.mean(anos_sem_pesca_ind)
aspi_std = np.sqrt( np.sum((anos_sem_pesca_ind - aspi_media)**2)/5  )
aspi_normalizado = (anos_sem_pesca_ind - aspi_media)/aspi_std

#Pegando o modelo e seus pesos salvos
model = keras.models.load_model('/home/andre/Área de Trabalho/Projeto da pesca/modelo_projeto_pesca.keras') 
model.load_weights('/home/andre/Área de Trabalho/Projeto da pesca/Lulas/pesos pesca ind/.weights.h5')

#Previsões
prev_geral = model.predict(ano_normalizado)*pi_std + pi_media
prev_1990_1995 = model.predict(aspi_normalizado)*pi_std + pi_media

#Plot dos dados e previsões
f1=plt.figure(1).add_subplot()
f1.scatter(ano,pesca_ind,label='Dados disponíveis')
f1.plot(ano,prev_geral,color='orange', label='Fit da rede' )
f1.scatter(anos_sem_pesca_ind,prev_1990_1995, color='red', label='Previsões de 1990 ate 1995')

#Print das previsões
print('Previsçoes de 1990 ate 1995:', prev_1990_1995)


plt.legend()
plt.show()