"""
Tamar Ervin
Date: December 8, 2021

Looking at Pyr data and opening the .tel files

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import time
from source import pyr_funcs as pf
from sklearn.cluster import KMeans


# get the files
dir_path = "/Users/tervin/pyr_classifier/pyr_data"
files = os.listdir(dir_path)
all_files = sorted(files)

# parse files
# days, full_flux, noon_flux = pf.parse_files(files, dir_path)
dates, flux, voltage = [], [], []
for fl in files:
    file = os.path.join(dir_path, fl)
    d, f, v = pf.read_pyr(file)
    dates.append(d)
    flux.append(f)
    voltage.append(v)

use_dates = [item for sublist in dates for item in sublist]
use_flux = [item for sublist in flux for item in sublist]
use_flux = np.array(use_flux)

# split dates and times
dates_list, times_list = [], []
for d in use_dates:
    dates_list.append(d.date())
    times_list.append(d.time())

# parse dates
unique_days = np.unique(dates_list)
unique_days = sorted(unique_days)
full_flux, noon_flux, days = [], [], []
unique_days = [unique_days[x] for x in np.arange(0, len(unique_days), 12)]
for i, d in enumerate(unique_days):
    d_use = np.isin(dates_list, d)
    d_times = np.array(times_list)[d_use]
    if len(d_times) == 3600*24:
        print(i)
        d_flux = use_flux[d_use]
        t_use = np.logical_and(d_times > time(hour=9, minute=30), d_times < time(hour=15, minute=30))
        days.append(d)
        full_flux.append(d_flux)
        noon_flux.append(d_flux[t_use])

# create model for ideal days
# default location parameters - NEID spectrometer (Kitt Peak, Tuscon AZ)
lat = 32.2
lon = -111
tz = 'US/Arizona'
elevation=735
name='Tucson'

# create TSI model
full_model, noon_model = pf.tsi_model(days, full_flux, lat, lon, tz, elevation, name)

# calculate statistical parameters
stat_params = pf.stat_parameters(days, full_flux, noon_flux, full_model)

# use elbow method to determine best number of clusters
sum_of_squared_distances = []
K = range(2,15)
for k in K:
    k_means = KMeans(n_clusters=k)
    model = k_means.fit(stat_params)
    sum_of_squared_distances.append(k_means.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Variance')
plt.title('Elbow method to determine optimal k value')
plt.show()
plt.savefig('/Users/tervin/pyr_classifier/images/elbow_method.png')

# create actual model
kmeans = KMeans(n_clusters=5, random_state=0).fit(stat_params)
labels = kmeans.labels_

# plot labels
plt.scatter(np.array(days), np.array(labels), color='lightblue', edgecolor='k', linewidths=0.8)
plt.xlabel('Days')
plt.ylabel('Classes')
plt.title('Trained Classifier')
plt.show()

# plot clustering
zero = np.where(labels == 0)
one = np.where(labels == 1)
two = np.where(labels == 2)
three = np.where(labels == 3)
four = np.where(labels == 4)
# five = np.where(labels == 5)
x = np.arange(0, len(days))
plt.scatter(stat_params[:,0][zero], stat_params[:,1][zero], color='red', edgecolors='k', linewidths=0.8, label='Label Zero')
plt.scatter(stat_params[:,0][one], stat_params[:,1][one], color='orange', edgecolors='k', linewidths=0.8, label='Label One')
plt.scatter(stat_params[:,0][two], stat_params[:,1][two], color='green', edgecolors='k', linewidths=0.8, label='Label Two')
plt.scatter(stat_params[:,0][three], stat_params[:,1][three], color='blue', edgecolors='k', linewidths=0.8, label='Label Three')
plt.scatter(stat_params[:,0][four], stat_params[:,1][four], color='purple', edgecolors='k', linewidths=0.8, label='Label Four')
# plt.scatter(stat_params[:,0][five], stat_params[:,1][five], color='pink', edgecolors='k', linewidths=0.8, label='Label Five')
plt.xlabel('Residual Standard Deviation')
plt.ylabel('Solar Noon Mean')
plt.legend()
plt.show()
plt.savefig('/Users/tervin/pyr_classifier/images/clutering_visualized.png')

# plot flux from each cluster
plot_flux = np.array(full_flux)
plt.plot(plot_flux[zero][0], color='red', label='Label Zero')
plt.plot(plot_flux[one][0], color='orange', label='Label One')
plt.plot(plot_flux[two][0], color='green', label='Label Two')
plt.plot(plot_flux[three][0], color='blue', label='Label Three')
plt.plot(plot_flux[four][0], color='purple', label='Label Four')
plt.xlabel('Time (s)')
plt.ylabel('Flux')
plt.legend()
plt.show()
plt.savefig('/Users/tervin/pyr_classifier/images/example_plots.png')

# plot all of them to look
ll = [two, four, zero, one, three]
col = ['green', 'purple', 'red', 'orange', 'blue']
for i, l in enumerate(ll):
    for j in l[0]:
        plt.plot(plot_flux[j], color=col[i])
        plt.title(str(days[j]))
        plt.show()

# trying to cluster just the good data
good_data = np.where(labels == 2)
good_params = stat_params[good_data]
kmeans_good = KMeans(n_clusters=2, random_state=0).fit(good_params)
labels_good = kmeans_good.labels_

# plot clustering
zero = np.where(labels_good == 0)
one = np.where(labels_good == 1)
# two = np.where(labels == 2)
# three = np.where(labels == 3)
# four = np.where(labels == 4)
# five = np.where(labels == 5)
x = np.arange(0, len(days))
plt.scatter(good_params[:,0][zero], good_params[:,1][zero], color='red', edgecolors='k', linewidths=0.8, label='Label Zero')
plt.scatter(good_params[:,0][one], good_params[:,1][one], color='orange', edgecolors='k', linewidths=0.8, label='Label One')
# plt.scatter(stat_params[:,0][two], stat_params[:,1][two], color='green', edgecolors='k', linewidths=0.8, label='Label Two')
# plt.scatter(stat_params[:,0][three], stat_params[:,1][three], color='blue', edgecolors='k', linewidths=0.8, label='Label Three')
# plt.scatter(stat_params[:,0][four], stat_params[:,1][four], color='purple', edgecolors='k', linewidths=0.8, label='Label Four')
# plt.scatter(stat_params[:,0][five], stat_params[:,1][five], color='pink', edgecolors='k', linewidths=0.8, label='Label Five')
plt.xlabel('Residual Standard Deviation')
plt.ylabel('Solar Noon Mean')
plt.legend()
plt.show()

# use pickle to save and load model
import pickle

# save model
pickle.dump(kmeans, open("../model.pkl", "wb"))

# load model
model = pickle.load(open("../model.pkl", "rb"))
model.predict(stat_params)


#
# pyr_file = '/Users/tervin/sdo_hmi_rvs/pyr_data/neid_ljpyrohelio_chv0_20210405.tel'
# pyr_file = '/Users/tervin/sdo_hmi_rvs/pyr_data/neid_ljpyrohelio_chv0_20210412.tel'
# pyr_data = pd.read_csv(pyr_file, sep='\s+', \
#     names=["timestamp", "voltage", "solar_flux"], \
#     parse_dates=["timestamp"], index_col="timestamp")
# flux = pyr_data.solar_flux
# voltage = pyr_data.voltage
# dates = pyr_data.index
#
# # plotting it to look at data
# xdata = dates
# plt.plot(xdata, flux, color='lavender')
# plt.xticks(rotation=45, ha="right")
# plt.xlabel('Time')
# plt.ylabel(r'$\rm Solar \/\rm Flux \/\rm [W m^{-1}]$')
# plt.title('Pyr Data Week of April 12, 2021')
# plt.show()
#
# # split up by date
# dates_list = []
# times_list = []
# for d in dates:
#     dates_list.append(str(d.year) + '-' + str(d.month) + '-' + str(d.day))
#     times_list.append(str(d.hour) + ':' + str(d.minute) + ':' + str(d.second))
#
# unique_days = np.unique(dates_list)
# unique_days = np.sort(unique_days)
# for d in unique_days:
#     d_use = np.isin(dates_list, d)
#     d_times = np.array(times_list)[d_use]
#     date_flux = flux[d_use]
#     plt.plot(d_times, date_flux, color='lightblue')
#     plt.xlabel('Time')
#     plt.ylabel(r'$\rm Solar \/\rm Flux \/\rm [W m^{-1}]$')
#     plt.title('Pyr Data: ' + str(d))
#     plt.xticks(rotation=45, ha="right")
#     plt.show()
#
# # use k-means clustering to classify -- not so good
# from sklearn.cluster import KMeans
#
# X = np.zeros((len(unique_days), int(len(flux)/len(unique_days))))
# for i, d in enumerate(unique_days):
#     d_use = np.isin(dates_list, d)
#     d_times = np.array(times_list)[d_use]
#     date_flux = flux[d_use]
#     X[i] = date_flux
#
# kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
# labels = kmeans.labels_
#
# for i, d in enumerate(unique_days):
#     d_use = np.isin(dates_list, d)
#     d_times = np.array(times_list)[d_use]
#     date_flux = flux[d_use]
#     plt.plot(d_times, date_flux, color='lightblue')
#     plt.xlabel('Time')
#     plt.xticks([])
#     plt.ylabel(r'$\rm Solar \/\rm Flux \/\rm [W m^{-1}]$')
#     plt.title('Pyr Data: ' + str(d) + '\n Classification: ' + str(labels[i]))
#     plt.xticks(rotation=45, ha="right")
#     plt.show()