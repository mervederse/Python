#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:45:37 2020

@author: yasinkutuk
"""

import numpy as np
import pandas as pd
pd.set_option('display.precision', 2)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%config InlineBackend.figure_format = 'retina'


plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['image.cmap'] = 'viridis'



df = pd.read_csv('/media/video_games_sales.csv', sep = ',')


df = df.dropna() # NA, NaN


df['User_Score'] = df['User_Score'].astype('float64')
df['Year_of_Release'] = df['Year_of_Release'].astype('int64')
df['User_Count'] = df['User_Count'].astype('int64')
df['Critic_Count'] = df['Critic_Count'].astype('int64')


df[ [i for i in df.columns if 'Sales' in i] + 
['Year_of_Release']].groupby('Year_of_Release').sum().plot(kind='bar', rot=90)


# Scatterplotting (Saçılım Grafiği)
sns.pairplot(df[['Global_Sales', 'User_Score', 'Critic_Score', 'User_Count']])

# Dağılım Grafiği
sns.distplot(df['Critic_Score'])

df['Critic_Score'].describe()
oyunlarin_ortalama_skoru1 = df['Critic_Score'].mean()
oyunlarin_ortalama_skoru2 = df['User_Score'].mean()*10


# JointPlot (Birleşim Grafiği)
sns.jointplot('Critic_Score', y= 'User_Score', data=df, kind='scatter')

# Heat Map
heat_pivot_table = df.pivot_table(index='Platform',
               columns = 'Genre',
               values = 'Global_Sales',
               aggfunc=sum).fillna(0).applymap(float)

sns.heatmap(heat_pivot_table)

# Sütunların yıllar olduğu ve platformun y ekseninde yer aldığı bir pivot 
# table üzerinden Wii platformunda en çok oyunun hangi yılda satıldığını 
# heatmap aracılığı ile bulunuz.


