"""
Tamar Ervin
Date: December 8, 2021

Looking at Pyr data and opening the .tel files

timestamps are in UTC, convert to UTC-7 by subtracting 7 hours
collect solar data: 9:30 AM to 3:30 UTC-7 (only look at flux in this frame)
"""

import os
from datetime import datetime, timezone, timedelta, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# opening the pyr data

pyr_file = '/pyr_data/neid_ljpyrohelio_chv0_20210405.tel'
pyr_data = pd.read_csv(pyr_file, sep='\s+', names=["timestamp", "voltage", "solar_flux"], parse_dates=["timestamp"],
                       index_col="timestamp")
flux = pyr_data.solar_flux
voltage = pyr_data.voltage
dates = pyr_data.index
dates = [d - timedelta(hours=7) for d in dates]

# split up by date
dates_list = []
times_list = []
for d in dates:
    day = str(d.day)
    dates_list.append(d.date())
    times_list.append(d.time())

# plotting it to look at data
xdata = dates
plt.plot(xdata, flux, color='lavender')
plt.xticks(rotation=45, ha="right")
plt.xlabel('Time')
plt.ylabel(r'$\rm Solar \/\rm Flux \/\rm [W m^{-1}]$')
plt.title('Pyr Data Week of April 12, 2021')
plt.show()

# split data by date and save
unique_days = np.unique(dates_list)
unique_days = sorted(unique_days)
date_flux = []
sol_flux = []
for i, d in enumerate(unique_days):
    d_use = np.isin(dates_list, d)
    d_times = np.array(times_list)[d_use]
    d_flux = flux[d_use]
    t_use = np.logical_and(d_times > time(hour=9, minute=30), d_times < time(hour=15, minute=30))
    date_flux.append(d_flux)
    sol_flux.append(d_flux[t_use])

for i, d in enumerate(unique_days):
    plt.plot(date_flux[i], color='lightblue')
    plt.xlabel('Time')
    plt.ylabel(r'$\rm Solar \/\rm Flux \/\rm [W m^{-1}]$')
    plt.title('Pyr Data: ' + str(d))
    plt.xticks([])
    plt.show()

# split data into training and test data
from sklearn.model_selection import train_test_split

y = [3, 4, 0, 4, 2, 1, 4]
n_features = 1
trainX, testX, trainy, testy = train_test_split(np.array(X), y, test_size=0.33, random_state=42)
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], n_features))

# create CNN
from keras.layers import *
from keras.models import Sequential

# array is 6*3600 by 1 (one point for each second of taking data and one flux value)
n_steps = 6 * 3600
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(trainX, trainy, epochs=10, verbose=0)
# demonstrate prediction
testX = testX.reshape((testX.shape[0], testX.shape[1], n_features))
yhat = model.predict(testX, verbose=0)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3


# split into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# verbose, epochs, batch_size = 0, 10, 24
# n_timesteps, n_features, n_outputs = 24*3600, 1, 3
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# # model.add(Dense(24*3600, activation='relu'))
# model.add(Dense(n_outputs, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # fit network
# model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# # evaluate model
# _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

### using pvlib package to make output power curves

from pvlib import pvsystem, location, modelchain
import pandas as pd
import matplotlib.pyplot as plt

class DualAxisTrackerMount(pvsystem.AbstractMount):
    def get_orientation(self, solar_zenith, solar_azimuth):
        # no rotation limits, no backtracking
        return {'surface_tilt': solar_zenith, 'surface_azimuth': solar_azimuth}


loc = location.Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
array = pvsystem.Array(
    mount=DualAxisTrackerMount(),
    module_parameters=dict(pdc0=1, gamma_pdc=-0.004, b=0.05),
    temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3))
system = pvsystem.PVSystem(arrays=[array], inverter_parameters=dict(pdc0=3))
mc = modelchain.ModelChain(system, loc, spectral_model='no_loss')

times = pd.date_range(start='2021-04-07', end='2021-04-08', freq='1S', tz=loc.tz)
weather = loc.get_clearsky(times)
mc.run_model(weather)

df = 3
x = np.arange(0, 86400)
# plt.plot(x, weather.ghi.values[:-1], color='lightgreen', label='Simulated GHI Data')
plt.plot(x, weather.dni.values[:-1] + np.min(date_flux[df]), color='orchid', label='Simulated DNI Data')
# plt.plot(x, weather.dhi.values[:-1], color='orchid', label='dhi Data')
# plt.plot(x, mc.results.ac.values[:-1]*1000, color='orchid', label='Simulated Data')
plt.plot(x, date_flux[df], color='lightblue', label='Actual Data')
plt.ylabel(r'Irradiance $\rm W/m^2$')
plt.xlabel('Time (s): 04/07/2021')
plt.legend()
plt.show()


#### for many days
from pvlib import pvsystem, location, modelchain, clearsky
import pandas as pd
import matplotlib.pyplot as plt

# plot for comparison
class DualAxisTrackerMount(pvsystem.AbstractMount):
    def get_orientation(self, solar_zenith, solar_azimuth):
        # no rotation limits, no backtracking
        return {'surface_tilt': solar_zenith, 'surface_azimuth': solar_azimuth}

lat = 32.2
lon = -111
loc = location.Location(lat, lon, 'US/Arizona', 700, 'Tucson')

for i, d in enumerate(unique_days):
    times = pd.date_range(start=d, end=unique_days[i] + timedelta(days=1), freq='1S', tz=loc.tz)
    # get turbidity
    turbidity = clearsky.lookup_linke_turbidity(times, lat, lon, interp_turbidity=False)
    weather = loc.get_clearsky(times, linke_turbidity=turbidity)
    x = np.arange(0, 86400)
    if len(date_flux[i]) == len(x):
        plt.plot(x, weather.dni.values[:-1] + np.min(date_flux[i]), color='orchid', label='Simulated DNI Data')
        plt.plot(x, date_flux[i], color='lightblue', label='Actual Data')
        plt.ylabel(r'Irradiance $\rm W/m^2$')
        plt.xlabel('Time (s): ' + str(d))
        plt.legend()
        plt.show()