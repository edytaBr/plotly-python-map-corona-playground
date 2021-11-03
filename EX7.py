#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:40:27 2021

@author: edyta
"""
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

ratings = pd.read_csv('ratings.csv', decimal=',')
item = pd.read_csv('item_based.csv', decimal=',')
coll = ratings - ratings.mean()
coll.drop(['Unnamed: 0'], axis=1, inplace=True)

corr = round(coll.corr(method = 'pearson'), 2)
collaborative = pd.read_csv('collaborative.csv',  decimal=",")

# %%
import pandas as pd
def openSheetsXlxs(name):
    file = pd.ExcelFile(name)
    listSheets = file.sheet_names
    for elem in listSheets:
         globals()[elem] = pd.read_excel(name, sheet_name=elem)
         globals()[elem].drop(['Unnamed: 0'], axis=1, inplace=True)

    return globals()[elem]

openSheetsXlxs("traffic_rules.xlsx")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
model = LinearRegression()


#Ordinal
train, test = train_test_split(rules_ordinal, test_size=0.2)
X = train.drop([train.columns[-1]], axis=1)
Y = train.drop(train.columns[0:-1], axis=1) #predict pass

X_test = train.drop([test.columns[-1]], axis=1)
Y_test = train.drop(test.columns[0:-1], axis=1) #predict pass
model.fit(X,Y)
prediction = model.predict(X_test)
print("Ordinary mse " , mean_squared_error(Y_test,prediction))

#Categorical
train, test = train_test_split(rules_categorical, test_size=0.2)
X = train.drop([train.columns[-1]], axis=1)
Y = train.drop(train.columns[0:-1], axis=1) #predict pass

X_test = train.drop([test.columns[-1]], axis=1)
Y_test = train.drop(test.columns[0:-1], axis=1) #predict pass
model.fit(X,Y)
prediction = model.predict(X_test)
print("Categorical mse " , mean_squared_error(Y_test,prediction))

# %%
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

c = pd.read_csv('time_series_covid_19_confirmed.csv', decimal='.')
d = pd.read_csv('time_series_covid_19_deaths.csv', decimal='.')

coor = c.drop_duplicates(subset=['Country/Region'], keep='last')
coor = coor[['Long', 'Lat']]

c["Sum"] = c.sum(axis=1)
d["SumD"] = d.sum(axis=1)

c = c[['Country/Region', 'Long', 'Lat', 'Sum']]
d = d[['Country/Region', 'Long', 'Lat', 'SumD']]

data = c.groupby('Country/Region').sum()
data2 = d.groupby('Country/Region').sum()
data2 = data2.reset_index()
data2.drop(data2[data2['Country/Region'] == 'Kiribati'].index, inplace=True)
data = data.reset_index()

data['Long'] = coor['Long'].values
data['Lat'] = coor['Lat'].values
data['Lat'] = coor['Lat'].values

data = data.reset_index()
data.drop(data[data['Country/Region'] == 'Kiribati'].index, inplace=True)
data['Deaths'] = data2['SumD'].values
px.set_mapbox_access_token(open("a.mapbox_token").read())
df = data
fig = px.scatter_mapbox(df, lat="Lat", lon="Long", size="Sum",color = "Deaths",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=100, zoom=1.5)
fig.show()