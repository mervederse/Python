# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:40:37 2020

@author: mvred
"""

# KNN En Yakın Komşular
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
%config InlineBackend.figure_format = 'retina'


# İki alt veri oluşturuyoruz
np.random.seed(17)
train_data = np.random.normal(size=(100,2))
train_labels = np.zeros(100)

train_data = np.r_[train_data, np.random.normal(size=(100,2),loc=2)]
train_labels = np.r_[train_labels,np.ones(100)]

plt.figure(figsize=(10,8))
plt.scatter(train_data[:,0], train_data[:,1], c=train_labels, s=100,
            cmap='autumn'); #s ile etiket koyuyoruz -cmap ile renklendiriyoruz.
plt.plot(range(-2,5), range(4,-3,-1))

from sklearn.tree import DecisionTreeClassifier
def getgrid(data):   #getgrid bir atlama fonksiyonudur.
    x_min, x_max = data[:,0].min() - 1, data[:,0].max() + 1
    y_min, y_max = data[:,1].min() - 1, data[:,1].max() + 1
    return np.meshgrid(np.arange(x_min,x_max,0.01), np.arange(y_min,y_max,0.01))

#ağaç grafiği oluşturuyoruz.
clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17) 

clf_tree.fit(train_data, train_labels)

xx,yy = getgrid(train_data)
tahmini_degerler= clf_tree.predict(np.c_[xx.ravel(), 
                                   yy.ravel()]).reshape(xx.shape)


plt.pcolormesh(xx,yy, tahmini_degerler, cmap='autumn')  #ayırma fonksiyonu
plt.scatter(train_data[:,0], train_data[:,1], c=train_labels, s=100,
            cmap='autumn', edgecolor='blue')

# Gözetimli ve Gözetimsiz Öğrenme Farkı
Supervised (Gözetimli)
Adı Maaş Cinsiyet
A 2800 Erkek
B 2500 Erkek
C 4100 Kadın

....

Unsupervised (Gözetimsiz)
Adı Maaş
A 2800 
B 2500
C 4100 

