import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import median_absolute_error,r2_score
import tensorflow as tf


dataset=pd.read_csv('Evolution_DataSets.csv')



#veri sayısı
print(f"data count: {dataset.shape[0]}")

#eksik verilerin kontrolü
print("Eksik veri:",dataset.isnull().sum()) #eksik verileri inceleyecek kod


print("eksik verileri doldur",dataset.fillna(0,inplace=True))
print(dataset.isnull().sum())
print("null değerler silindi",dataset.dropna(inplace = True))#(inplace = True) )#inplace=true kalıcı değişiklik
print(dataset.isnull().sum())
#print("eksik veriler kaldirildi",dataset.to_string())
print("count:",dataset.count())


print("**************************")


# Kategorik değişkenlerin kontrolü



print(dataset.describe())



#type'ına göre dağılım

x=dataset.groupby(['Anatomy'])['Anatomy'].count()
y=len(dataset)
r=((x/y)).round(2)
mf_ratio = pd.DataFrame(r).T
print("Anatomy",r) ##type'a göre dağılım
print(mf_ratio)
print('*****')

#Eğitim ve test veri setlerinin ayrılması


print(dataset.iloc[1:5]) #belirlediğim indexlerdeki verileri çektim.
print("********")
X=dataset.iloc[:,:-1].values
type(X)
Y=dataset.iloc[:,:-1].values
type(Y)
Y

#Veri setini train ve test olarak ayırdık
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0) #test kısmına tüm verinin 30%'sini ayır.

print("test",X_train) 


#ölçeklendirme
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#X_train=scaler.fit_transform(X_train) #ölçeklendir ve dönüştür
#X_test=scaler.fit_transform(X_test)
print(X_test)
#Y_train_train=scaler.fit_transform(Y_train.reshape(-1,1))
#Y_test_train=scaler.fit_transform(Y_test.reshape(-1,1)) #çıktılarımı -1 ile 1 arası şekillendir

#Ağın oluşturulması
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


classification=Sequential()

#Dense ile katmanlar arasında nöron ya da düğümlerin geçişlerini sağlar. Bir başka deyişle, bir katmandan aldığı nöronları bir sonraki katmana girdi olarak bağlanmasını sağlar.
#ilk gizli katman

classification.add(Dense(units=8,activation='relu',input_dim=5))
#ikinci gizli katman
classification.add(Dense(units=8,activation='relu'))
#üçüncü katman
classification.add(Dense(units=8,activation='relu'))
#Çıktı katmanı
classification.add(Dense(units=1,activation='sigmoid'))
#Derleme
classification.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #adam=adaptik öğrenme fonksiyonu
#model özeti
classification.summary() 
#Eğitim
classification.fit(X_train, Y_train,batch_size=10, epochs=100)

#test seti üzerindeki tahminler
Y_pred_Ysa=classification.predict(X_test)
#performans değerlendirmesi
r2_score=(Y_test,Y_pred_Ysa)
mean_absolute_error(Y_test,Y_pred_Ysa)
mean_squared_error(Y_test,Y_pred_Ysa)
median_absolute_error(Y_test,Y_pred_Ysa)

#Ağırlıklar
for i in classification.layers:
    ilk_gizli_katman=classification.layers[0].get_weights()
    ikinci_gizli_katman=classification.layers[1].get_weights()
    cikti_gizli_katman=classification.layers[2].get_weights()
#Ters ölçeklendirme
    olceklendirme_ters=scaler.inverse_transform(Y_pred_Ysa.reshape(-1,1))