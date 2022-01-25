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
from scipy import stats


def read_pyr(pyr_file):
    """
    Function to take in .tel file with Pyr data and return relevant information.

    :param str pyr_file: file path
    :return:
        - dates: list of dates
        - flux: list of flux values
        - voltage: list of voltage values
    :rtype:
        - List[DateTime]
        - List[float]
        - List[float]
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
    :param List[DateTime] dates: list of dates
    :param List[float] flux: list of flux values
    :return:
        - days - list of days
        - full_flux - list of full flux values
        - noon_flux - list of solar noon flux values
    :rtype:
        - List[str]
        - List[float]
        - List[float]
    """

    # split dates into days and times
    dates_list, times_list = [], []
    for d in dates:
        dates_list.append(d.date())
        times_list.append(d.time())

    # turn lists into arrays
    dates_list = np.array(dates_list)
    times_list = np.array(times_list)

    # get unique dates
    unique_days = np.unique(dates_list)
    unique_days = sorted(unique_days)

    # parse dates
    full_flux, noon_flux, days = [], [], []
    for i, d in enumerate(unique_days):
        d_use = np.where(d == dates_list)
        d_times = times_list[d_use]
        if len(d_times) == 3600*24:
            d_flux = flux[d_use]
            t_use = np.logical_and(d_times > time(hour=9, minute=30), d_times < time(hour=15, minute=30))
            noon_use = np.logical_and(d_times > time(hour=11, minute=30), d_times < time(hour=12, minute=30))
            days.append(d)
            full_flux.append(d_flux)
            noon_flux.append(d_flux[noon_use])

    return days, full_flux, noon_flux


def parse_files(files, dir_path):
    """
    function to parse files and return dates, full flux, noon flux
    :param List[str] files: list of files
    :param str dir_path: directory path
    :returns:
        - days - list of days
        - full_flux - list of full flux values
        - noon_flux - list of solar noon flux values
    :rtype:
        - List[str]
        - List[float]
        - List[float]
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


def tsi_model(days, full_flux, lat=32.2, lon=-111, tz='US/Arizona', elevation=735, name='Tucson'):
    """
    function to create TSI model for specific days and location

    :param List[str] days: list of unique days
    :param List[float] full_flux: list of flux arrays corresponding to each date
    :param float lat: latitude
    :param float lon: longitude
    :param str tz: Timezone
    :param int elevation: elevation value
    :param str name: location name
    :returns:
        - full_model - full flux model
        - noon_model - solar noon flux model
    :rtype:
        - List[float]
        - List[float]
    """

    # location using pvlib package -- latitude, longitude, timezone, altitude, latitude
    loc = location.Location(lat, lon, tz, elevation, name)

    full_model, noon_model = [], []

    for i, d in enumerate(days):
        # create times array for model
        times = pd.date_range(start=d, end=d + timedelta(days=1), freq='1S', tz=loc.tz)
        # times = times

        # get turbidity
        turbidity = clearsky.lookup_linke_turbidity(times, lat, lon, interp_turbidity=False)

        # create model
        weather = loc.get_clearsky(times, linke_turbidity=turbidity)

        # scale model
        mod = weather.dni.values + np.min(full_flux[i])

        # create full model and solar noon model
        times = times.time
        t_use = np.logical_and(times >= time(hour=9, minute=30), times < time(hour=15, minute=30))
        noon_use = np.logical_and(times >= time(hour=11, minute=30), times < time(hour=12, minute=30))
        full_model.append(mod[:-1])
        noon_model.append(mod[noon_use])

    return full_model, noon_model


def stat_parameters(days, full_flux, noon_flux, full_model, noon_model):
    """
    function to calculate statistical parameters for clustering
    :param List[str] days: list of unique days
    :param List[float] full_flux: list of full flux values
    :param List[float] noon_flux: list of solar noon flux values
    :param List[float] full_model: list of full flux model
    :param List[float] noon_model: list of solar noon flux model
    :return: list of statistical parameters for clustering
    :rtype: List[float]
    """

    # scaled residual standard deviation
    scal_f_std, scal_n_std, scal_n_mean, n_std = [], [], [], []
    for i, d in enumerate(days):
        f_res = full_flux[i] - (full_model[i] * 1.08)
        scal_f_std.append(np.std(f_res))
        f_res = noon_flux[i] - (noon_model[i] * 1.08)
        scal_n_std.append(np.std(f_res))
        scal_n_mean.append((np.mean(noon_flux[i])))
        n_std.append(np.std(noon_flux[i]))
    scal_f_std = np.array(scal_f_std)
    scal_n_mean = np.array(scal_n_mean)
    scal_n_std = np.array(scal_n_std)
    # s_std = np.array(s_std)
    n_std = np.array(n_std)

    # ([std values of scaled full flux array, mean values of scaled solar noon flux array])
    stat_params = np.column_stack((scal_f_std, n_std, scal_n_mean))

    return stat_params


def classify_days(days, stats_params, model_path, csv_name):
    """
    function to classify days using model and map labels to usable classes
    :param List[str] days: list of unique days
    :param List[float] stats_params: list of statistical parameters for clustering
    :param str model_path: path to clustering model
    :param str csv_name: path to save csv
    :return: dataframe with dates and labels
    :rtype: DataFrame
    """

    # load model
    model = pickle.load(open(model_path, "rb"))

    # use model for prediction
    labels = model.predict(stats_params)
    labels = np.array(labels)

    # map labels to correct class
    mapping = {5: 0, 1: 1, 0: 2, 4: 4, 3: 5}
    mapped = [mapping[i] for i in labels]
    mapped = np.array(mapped)

    # create array of dates and labels
    date_labels = np.column_stack((days, mapped))

    # save to csv
    df = save_to_csv(date_labels, csv_name)

    return df


def save_to_csv(date_labels, csv_name=None):
    """
    function to save labeled dates to csv
    :param date_labels: array with dates and corresponding labels
    :param str csv_name: path to save csv
    :return: dataframe with dates and labels
    :rtype: DataFrame
    """
    # create pandas dataframe
    df = pd.DataFrame(date_labels)

    # add column headers
    df.columns = ['date', 'label']

    # save csv
    if csv_name is not None:
        df.to_csv(csv_name)

    return df


def classify_files(dir_path, model_path, csv_name=None):
    """
    function for full classifier model

    :param str dir_path: path to directory with telemetry data
    :param str model_path: path to clustering model
    :param str csv_name: path to save csv
    :return: dataframe with dates and labels
    :rtype: DataFrame
    """

    # list out files
    files = os.listdir(dir_path)
    files = sorted(files)

    # parse files
    days, times_list, full_flux, noon_flux = parse_files(files, dir_path)

    # default location parameters - NEID spectrometer (Kitt Peak, Tuscon AZ)
    lat = 32.2
    lon = -111
    tz = 'US/Arizona'
    elevation = 735
    name = 'Tucson'

    # create TSI model
    full_model, noon_model = tsi_model(days, full_flux, lat, lon, tz, elevation, name)

    # calculate statistical parameters
    stats_params = stat_parameters(days, full_flux, noon_flux, full_model, noon_model)

    # classify dates based on trained model
    date_labels = classify_days(days, stats_params, model_path, csv_name)

    return date_labels


def classify_array(dates, flux, model_path, csv_name=None):
    """
    function for full classifier model

    :param List[DateTime] dates: list of dates
    :param List[float] flux: list of flux values
    :param str model_path: path to clustering model
    :param str csv_name: path to save csv
    :return: dataframe with dates and labels
    :rtype: DataFrame
    """

    # parse array
    days, full_flux, noon_flux = parse_dates(dates, flux)

    # default location parameters - NEID spectrometer (Kitt Peak, Tuscon AZ)
    lat = 32.2
    lon = -111
    tz = 'US/Arizona'
    elevation = 735
    name = 'Tucson'

    # create TSI model
    full_model, noon_model = tsi_model(days, full_flux, lat, lon, tz, elevation, name)

    # calculate statistical parameters
    stats_params = stat_parameters(days, full_flux, noon_flux, full_model, noon_model)

    # classify dates based on trained model
    date_labels = classify_days(days, stats_params, model_path, csv_name)

    return date_labels