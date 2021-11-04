#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:40:27 2021

@author: edyta
"""

import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.express as px
import plotly
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
                  color_continuous_scale='RdBu_r', size_max=100, zoom=1.5)
fig.show()
plotly.offline.plot(fig, filename='/home/edyta/git/plotly-python-map-corona-playground/index.html')