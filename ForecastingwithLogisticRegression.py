# -*- coding: utf-8 -*-
"""
Created on Sun May 17 02:32:32 2020

@author: mvred
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures#Features= değişkenler demek
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv('C://microchip_tests.txt', header=None,
                   names=('test1', 'test2', 'released'))

# Bağımlı ve Bağımsız Değişkenler
X = data.iloc[:,:2].values
y = data.iloc[:,2].values

#Grafik- köşeli parantezle koşulunu söylüyoruz.
#label ile nasıl okunması gerektiğini oluşturuyoruz
plt.scatter( X[y==1,0], X[y==1,1], c='blue', label= 'Başarılı')
# X veri setinden verileri al y verileri bir olsun
plt.scatter( X[y==0,0], X[y==0,1], c='red', label='Başarısız' )
# X veri setinden verileri al y verileri 0 olsun c= renk
plt.xlabel('Test 1') #x değikenin değikeni test1 olsun(grafikte x ekseninde gösterilir.)
plt.ylabel('Test 2')
plt.title('Çipsetlerin İki Testi. Başarılı=1')#0-9 a kadar olan rakamlar yazıyla yazılmalı başarılı olanlar bir diyoruz.
plt.legend();#görüntülüyoruz.



#clf = ayırma fonksiyonu.
#gridstep ile verilerin artışının kaç olduğunu söylüyoruz.
#x_min, x_max = X[:,0].min() - 0.10,-virgülden önce min virgülden sonra max değerini oluşturur- X[:,0].max() + 0.10
def ayirma(clf, X, y, grid_step= .01, poly_featurizer=None):
    x_min, x_max = X[:,0].min() - 0.10, X[:,0].max() + 0.10
    y_min, y_max = X[:,1].min() - 0.10, X[:,1].max() + 0.10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),\#meshgrid ile bir kordinasyon denkleminin kaç boyutlu olduğunu söylüyoruz.
                         np.arange(y_min,y_max, grid_step))#grid step ile eksenlerin artış aralığını tanımlıyoruz.
    # np.arange sınırlarını belirliyoruz.
   
#verinin x ve y değerlerinin yerlerini değiştiriyoruz.
tahmin = clf.predict(poly_featurizer.transfrom(np.c_[xx.ravel(),
                                                     yy.ravel()])) #ravel ile verilerin aralarını dolduruyoruz.
#tahminin boyutunu resshape ile değiştiriyoruz.
tahmin = tahmin.reshape(xx.shape)

#tahmini değerleri çizdir ve renklendirme
plt.contour(xx,yy, tahmin, cmap= plt.cm.Paired)

#Değişkenlerin Üstsel Kombinasyonları
poly = PolynomialFeatures(degree=7)#bağımsız değişkenin kendisini,karesini ve kaçıncı dereceden üstünü almam gerektiğine dair bir parametre
#x in kendisi karesi 7 ye kadar üssünü alarak deniycek
X_poly = poly.fit_transform(X)

X_poly.shape

#Tahmin Fonksiyonu
logit = LogisticRegression(C=1e-2, random_state=17) #C=1e-2 : 1 in yüzde 1 i demek
logit.fit(X_poly,y)#fonksiyonu bizim için tahmin eder
ayirma(logit, X, y, grid_step=.01, poly_featurizer=poly)#polly f. üstsel değişkenlerin tahmin edilmesi
print('Başarım yüzdesi:', round(logit.score(X_poly,y),2))

#logit.score(X_poly,y) kodu ile yüzde kaç başarılı bir tahmin olduğuna bakarız.

#Tahmin Fonksiyonu2
logit = LogisticRegression(C=1, random_state=17)#C parametresini 
logit.fit(X_poly,y)
ayirma(logit, X, y, grid_step=.005, poly_featurizer=poly)#grid step artış oranı demek
print('Başarım yüzdesi:', round(logit.score(X_poly,y),2))

#Tahmin Fonksiyonu3 1e5 yüz binde demek üstsel ifade yani
logit = LogisticRegression(C=1e5, random_state=17)#c parametresini değiştirmek başarı yüzdesini arttırdı.
#(iki kümeyi birbirinden ayırma başarı yüzdesidir)
logit.fit(X_poly,y)
ayirma(logit, X, y, grid_step=.005, poly_featurizer=poly)#♣logit yerine clf yazılabilir
print('Başarım yüzdesi:', round(logit.score(X_poly,y),2))
