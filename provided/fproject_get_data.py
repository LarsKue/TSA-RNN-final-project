# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:40:52 2022

@author: MaxIngo.Thurm
"""

#%%Settup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def running_mean(x, N):
    w = np.ones(N)/N
    if x.ndim > 1:
        res = []
        for i in range(x.shape[1]):
            res.append(np.convolve(w, x[:,i], 'valid'))
        return np.array(res).T
    else:
        return np.convolve(w, x, 'valid')

#%%Get raw data
# data = pd.read_csv(r"\\zisvfs12\Home\maxingo.thurm\Downloads\weather_data.csv")
data = pd.read_csv(r"https://data.open-power-system-data.org/weather_data/2020-09-16/weather_data.csv")

#%%Extract data for Finland, Germany and Greece 
keys = ['FI_temperature',
        'DE_temperature',
        'GR_temperature']

test_temp = data[keys]

#%%Get the last ten years [screw leap years]
ten_years = 24*365*10
tdf = test_temp[-ten_years:]

#%%Weekly Filter downsampling, only the first day of week is used to reduce data
week_filter = [True]*24 + [False]*24*6 #wekly mask
year_filter = week_filter*52 + [True]*24 # yearly mask 52 weeks plus one day
ten_years_filter = year_filter*10 #filter for ten years
daydf = tdf[ten_years_filter] 

#%%plotting with numy and maplotlib
npa = daydf.to_numpy()
#Moving average to see both trends
N = 24
mnpa = running_mean(npa, N)

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(npa)
axs[0].set_title('Data as given', fontsize=12)
axs[1].plot(mnpa)
axs[1].set_title('Yearly trend', fontsize=12)
axs[2].plot(npa[:-N+1]-mnpa)
axs[1].set_ylabel('Temp $[°C]$')
axs[2].set_title('Daily trend', fontsize=12)
axs[2].set_xlabel('Time $[h]$')


