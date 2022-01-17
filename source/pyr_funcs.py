"""
Tamar Ervin
Date: Jan 5, 2022

Function used to look at pyroheliometer data
and classify it.

"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta, time
from pvlib import location, clearsky


def read_pyr(pyr_file):
    """
    Function to take in .tel file with Pyr data and return relevant information.

    :param str pyr_file: file patth
    :return: dates: list of dates, flux: list of flux values, voltage: list of voltage values
    :rtype: DateTime, float, float
    """

    # read in file
    pyr_data = pd.read_csv(pyr_file, sep='\s+', names=["timestamp", "voltage", "solar_flux"], parse_dates=["timestamp"],
                           index_col="timestamp")

    # pull out relevant columns
    dates = np.array(pyr_data.index)
    flux = pyr_data.solar_flux
    voltage = pyr_data.voltage

    # convert time from UT to UTC-7 (local time)
    dates = pd.to_datetime(dates)
    dates = dates - timedelta(hours=7)

    return dates, flux, voltage


def parse_dates(dates, flux):
    """
    function to parse Pyr data by date and look at only when solar data is collected
    :param DateTime dates: list of dates
    :param float flux: list of flux values
    :return: days: list of dates, full_full: list of full flux values, noon_flux: list of solar noon flux values
    :rtype: DateTime
    :rtype: float
    :rtype: float
    """

    # split dates into days and times
    dates_list, times_list = [], []
    for d in dates:
        dates_list.append(d.date())
        times_list.append(d.time())

    unique_days = np.unique(dates_list)
    unique_days = sorted(unique_days)
    full_flux, noon_flux, days = [], [], []
    for i, d in enumerate(unique_days):
        d_use = np.isin(dates_list, d)
        d_times = np.array(times_list)[d_use]
        if len(d_times) == 3600*24:
            d_flux = flux[d_use]
            t_use = np.logical_and(d_times > time(hour=9, minute=30), d_times < time(hour=15, minute=30))
            days.append(d)
            full_flux.append(d_flux)
            noon_flux.append(d_flux[t_use])

    return days, full_flux, noon_flux


def parse_files(files, dir_path):
    """
    function to parse files and return dates, full flux, noon flux
    :param str files: list of files
    :param str dir_path: directory path
    :return: list of days
    :rtype: str
    :return: list of full flux values
    :rtype: float
    :return: list of solar noon flux values
    :rtype: float
    """

    dates, flux, voltage = [], [], []
    for fl in files:
        file = os.path.join(dir_path, fl)
        d, f, v = read_pyr(file)
        dates.append(d)
        flux.append(f)
        voltage.append(v)

    use_dates = [item for sublist in dates for item in sublist]
    use_flux = [item for sublist in flux for item in sublist]
    use_flux = np.array(use_flux)
    days, full_flux, noon_flux = parse_dates(use_dates, use_flux)

    return days, full_flux, noon_flux


def tsi_model(days, date_flux, lat=32.2, lon=-111, tz='US/Arizona', elevation=700, name='Tucson'):
    """
    function to create TSI model for specific days and location

    :param str days: list of unique days
    :param float date_flux: list of flux arrays corresponding to each date
    :param float lat: latitude
    :param float lon: longitude
    :return: full flux model
    :rtype: float
    :return: solar noon flux model
    :rtype: float
    """

    # location using pvlib package -- latitude, longitude, timezone, altitude, latitude
    loc = location.Location(lat, lon, tz, elevation, name)

    full_model, noon_model = [], []

    for i, d in enumerate(days):
        times = pd.date_range(start=d, end=d + timedelta(days=1), freq='1S', tz=loc.tz)
        # get turbidity
        turbidity = clearsky.lookup_linke_turbidity(times, lat, lon, interp_turbidity=False)
        weather = loc.get_clearsky(times, linke_turbidity=turbidity)
        times_list = []
        for d in times:
            times_list.append(d.time())
        mod = weather.dni.values + np.min(date_flux[i])
        times_comp = np.array(times_list)
        t_use = np.logical_and(times_comp >= time(hour=9, minute=30), times_comp < time(hour=15, minute=30))
        full_model.append(mod[:-1])
        noon_model.append(mod[t_use])

    return full_model, noon_model


def stat_parameters(days, full_flux, noon_flux, full_model):
    """
    function to calculate statistical parameters for clustering
    :param str days: list of unique days
    :param float full_flux: list of full flux values
    :param float noon_flux: list of solar noon flux values
    :param float full_model: list of full flux model
    :return: list of statistical parameters for clustering
    :rtype: float
    """

    # scaled residual standard deviation
    scal_f_std, scal_s_mean, good_flux, good_model = [], [], [], []
    for i, d in enumerate(days):
        # if len(full_flux[i]) == len(full_model[i]):
        good_flux.append(full_flux[i])
        good_model.append(full_model[i])
        f_res = full_flux[i] - (full_model[i] * 1.08)
        scal_f_std.append(np.std(f_res))
        scal_s_mean.append((np.mean(noon_flux[i])))
        # else:
        #     pass
    scal_f_std = np.array(scal_f_std)
    scal_s_mean = np.array(scal_s_mean)

    # ([std values of scaled full flux array, mean values of scaled solar noon flux array])
    stats_params = np.column_stack((scal_f_std, scal_s_mean))

    return stats_params


def classify_days(days, stats_params, model_path):
    """
    function to classify days using model and map labels to usable classes
    :param str days: list of unique days
    :param float stats_params: list of statistical parameters for clustering
    :param str model_path: path to clustering model
    :return: list labeled dates
    :rtype: int
    """

    # load model
    model = pickle.load(open(model_path, "rb"))

    # use model for prediction
    labels = model.predict(stats_params)
    labels = np.array(labels)

    # map labels to correct class
    mapping = {2: 0, 4: 1, 0: 2, 1: 3, 3: 4}
    mapped = [mapping[i] for i in labels]

    # create array of dates and labels
    date_labels = np.column_stack((days, mapped))

    return date_labels


def classifier_model(dir_path, model_path):
    """
    function for full classifier model

    :param str model_path: path to clustering model
    :param str dir_path: path to directory with telemetry data
    :return: list labeled dates
    :rtype: int
    """

    # list out files
    files = os.listdir(dir_path)
    files = sorted(files)

    # parse files
    days, full_flux, noon_flux = parse_files(files, dir_path)

    # default location parameters - NEID spectrometer (Kitt Peak, Tuscon AZ)
    lat = 32.2
    lon = -111
    tz = 'US/Arizona'
    elevation = 735
    name = 'Tucson'

    # create TSI model
    full_model, noon_model = tsi_model(days, full_flux, lat, lon, tz, elevation, name)

    # calculate statistical parameters
    stats_params = stat_parameters(days, full_flux, noon_flux, full_model)

    # classify dates based on trained model
    date_labels = classify_days(days, stats_params, model_path)

    return date_labels