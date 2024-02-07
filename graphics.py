# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:42:19 2020

@author: mvred
"""

import numpy as np
import pandas as pd
import matplotlib as plt
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

df= pd.read_csv('C://sales.csv',)

cift = df[df['Year_of_Release'].apply(lambda x:x % 2 ==0 )]

#Grafik 1
cift.head()
sns.scatterplot(x='Platform', y='Global_Sales', data=cift)
sns.scatterplot(x='Platform', y='Critic_Score', data=cift)
sns.scatterplot(x='Platform', y='User_Score', data=cift)

#Grafik 2
cift=cift.dropna()

cift['Platform']=cift['Platform'].astype('object')
cift['User_Score']=cift['User_Score'].astype('float64')
cift['Critic_Score'] = cift['Critic_Score'].astype('int64')


df2 = cift.sort_values(by='Global_Sales', ascending=False)


df2[[i for i in df.columns if 'Sales' in i]+
    ['Platform']].groupby('Platform').sum().plot()



#Grafik 3
#Heat Map
heat_pivot_table=cift.pivot_table(index='Platform',
               columns='Year_of_Release',
               values='Global_Sales',).fillna(0).applymap(float)
sns.heatmap(heat_pivot_table)


#Heat Map
heat_pivot_table=cift.pivot_table(index='Platform',
               columns='Year_of_Release',
               values='Critic_Score',).fillna(0).applymap(float)
sns.heatmap(heat_pivot_table)


heat_pivot_table=cift.pivot_table(index='Platform',
               columns='Year_of_Release',
               values='User_Score',).fillna(0).applymap(float)
sns.heatmap(heat_pivot_table)
















