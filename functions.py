""" Libraries """
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import torch
import random
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default="warn"
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



""" Functions """
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    return


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_normalized_data(normalize_settings={}, part="Train", outlier_constant=3, fit_test=False):

    default_normalize_settings = {
        "Mode"        : "Newer Module"  ,  # "together", "Module", "New Module", "Newer Module"
        "Module"      : "one-hot"       ,  # "one-hot", "original"
        "Date"        : "one-hot"       ,  # "range", "one-hot", "original"
        "Temp"        : "range"         ,  # "range", "max_abs", "original"
        "Temp_m"      : "range"         ,  # "range", "max_abs", "original"
        "Irradiance"  : "range"         ,  # "range", "max_abs", "original"
        "Irradiance_m": "range"         ,  # "range", "max_abs", "original"
        "I_m/I"       : "range"         ,  # "range", "max_abs", "original"
        "I/I_m"       : "range"         ,  # "range", "max_abs", "original"
        "I_m/I-min"   : "range"         ,  # "range", "max_abs", "original"
        "I/I_m-min"   : "range"         ,  # "range", "max_abs", "original"
        "Generation"  : "range"         ,  # "range", "max_abs", "original"
        "Capacity"    : "range"         ,  # "range", "max_abs", "original"
        "Lat"         : "range"         ,  # "range", "max_abs", "original"
        "Lon"         : "range"         ,  # "range", "max_abs", "original"
        "Angle"       : "range"         ,  # "range", "max_abs", "original"
    }

    for key in default_normalize_settings.keys():
        if key not in normalize_settings.keys():
            normalize_settings[key] = default_normalize_settings[key]

    assert normalize_settings["Mode"]    in ["together", "Module", "New Module", "Newer Module"]
    assert normalize_settings["Date"]    in ["range", "one-hot", "original"]
    assert normalize_settings["Module"]  in ["one-hot", "original"]
    assert part in ["All", "Train+Valid", "Train", "Valid", "Test"]

    train_data = pd.read_csv("data/train.csv")
    test_data  = pd.read_csv("data/test_filled.csv")
    data       = pd.concat((train_data, test_data))
    data.reset_index(inplace=True)
    data.drop("ID", inplace=True, axis=1)
    data.drop("index", inplace=True, axis=1)

    if normalize_settings["Date"] in ["range", "one-hot"]:
        datetime = pd.to_datetime(data["Date"])
        if normalize_settings["Date"] == "range":
            data["Year"]  = pd.Series((datetime.dt.year-2020)/2)
            data["Month"] = pd.Series((datetime.dt.month-1)/11)
            data["Day"]   = pd.Series((datetime.dt.day-1)/30)
        elif normalize_settings["Date"] == "one-hot":
            for year in list(range(2020, 2022+1)):
                data[f"Year-{year}"]   = pd.Series(datetime.dt.year==year, dtype=int)
            for month in list(range(1, 12+1)):
                data[f"Month-{month}"] = pd.Series(datetime.dt.month==month, dtype=int)
            for day in list(range(1, 31+1)):
                data[f"Day-{day}"]     = pd.Series(datetime.dt.day==day, dtype=int)
        data.drop("Date", inplace=True, axis=1)
    

    # Customized
    data["I_m/I"] = data["Irradiance_m"] / data["Irradiance"]
    data["I/I_m"] = data["Irradiance"] / data["Irradiance_m"]


    standard_normalize_keys = ["Generation", "Temp", "Temp_m",
                               "Irradiance", "Irradiance_m", 
                               "I_m/I", "I/I_m",
                               "I_m/I-min", "I/I_m-min",
                               "Capacity", "Lat", "Lon", "Angle"]


    if normalize_settings["Mode"] != "together":

        if normalize_settings["Mode"] == "Module":
            MODULES = ["MM60-6RT-300", "SEC-6M-60A-295", "AUO PM060MW3 320W", "AUO PM060MW3 325W"]
        elif normalize_settings["Mode"] == "New Module":
            MODULES = ["AUO PM060MW3 320W_24.04_120.52", "AUO PM060MW3 320W_24.06_120.47",
                       "AUO PM060MW3 320W_24.07_120.47", "AUO PM060MW3 320W_24.07_120.48",
                       "AUO PM060MW3 320W_24.08_120.52", "AUO PM060MW3 325W_24.08_120.52",
                       "AUO PM060MW3 320W_24.08_120.5", "AUO PM060MW3 320W_24.09_120.52",
                       "AUO PM060MW3 320W_24.107_120.44", "SEC-6M-60A-295_24.98_121.03",
                       "MM60-6RT-300_25.03_121.08", "MM60-6RT-300_25.11_121.26"]
        elif normalize_settings["Mode"] == "Newer Module":
            MODULES = ["AUO PM060MW3 320W_246.4_24.107_120.44_4.63", "AUO PM060MW3 320W_267.52_24.08_120.52_-2.13",
                       "AUO PM060MW3 320W_278.4_24.09_120.52_0.0", "AUO PM060MW3 320W_314.88_24.06_120.47_0.0",
                       "AUO PM060MW3 320W_352.0_24.07_120.47_0.0", "AUO PM060MW3 320W_492.8_24.107_120.44_4.63",
                       "AUO PM060MW3 320W_498.56_24.04_120.52_2.21", "AUO PM060MW3 320W_498.56_24.04_120.52_2.21_temp=60",
                       "AUO PM060MW3 320W_99.2_24.08_120.5_1.76", "AUO PM060MW3 320W_99.84_24.07_120.48_10.35",
                       "AUO PM060MW3 325W_343.2_24.08_120.52_-2.62", "MM60-6RT-300_438.3_25.11_121.26_-160.0",
                       "MM60-6RT-300_498.6_25.03_121.08_-95.0", "MM60-6RT-300_499.8_25.11_121.26_22.0",
                       "SEC-6M-60A-295_283.2_24.98_121.03_-31.0"]
        module = []
        for md, cp, lat, lon, ag, tm in zip(
            data["Module"].values,
            data["Capacity"].values,
            data["Lat"].values,
            data["Lon"].values,
            data["Angle"].values,
            data["Temp_m"].values,
        ):
            if normalize_settings["Mode"] == "Module":
                nm = md
            elif normalize_settings["Mode"] == "New Module":
                nm = f"{md}_{lat}_{lon}"
            elif normalize_settings["Mode"] == "Newer Module":
                nm = f"{md}_{cp}_{lat}_{lon}_{ag}"
                if nm=="AUO PM060MW3 320W_498.56_24.04_120.52_2.21" and tm==60:
                    nm += "_temp=60"
            module.append(nm)
        data[normalize_settings["Mode"]] = pd.Series(module)
        if normalize_settings["Mode"] != "Module":
            data.drop("Module", inplace=True, axis=1)

        module_data = {}
        for module in MODULES:
            module_data[module] = data[data[normalize_settings["Mode"]]==module]
            for key in standard_normalize_keys:
                assert normalize_settings[key] in ["range", "max_abs", "original"]

                module_data_key         = module_data[module][key]
                module_data_key_nan     = module_data_key[module_data_key.isna()]
                module_data_key_not_nan = module_data_key[module_data_key.notna()]

                # Process outliers
                if key in ["Generation", "Temp_m", "Irradiance_m", "I_m/I", "I/I_m"]:  # , "I_m/I-min", "I/I_m-min"]:
                    if key not in ["I_m/I", "I/I_m"]:  # , "I_m/I-min", "I/I_m-min"]:
                        is_outlier = (module_data_key_not_nan-module_data_key_not_nan.mean()).abs() > outlier_constant*module_data_key_not_nan.std()

                    # Mannually add outliers
                    if key=="Generation":
                        if module=="AUO PM060MW3 320W_246.4_24.107_120.44_4.63":
                            is_outlier = is_outlier | (module_data_key_not_nan > 1800)
                        elif module=="AUO PM060MW3 320W_314.88_24.06_120.47_0.0":
                            is_outlier = is_outlier | (module_data_key_not_nan > 2100)
                        elif module=="AUO PM060MW3 320W_492.8_24.107_120.44_4.63":
                            is_outlier = is_outlier | (module_data_key_not_nan > 3700)
                        elif module=="AUO PM060MW3 320W_99.2_24.08_120.5_1.76":
                            is_outlier = is_outlier | (module_data_key_not_nan > 630)
                        elif module=="MM60-6RT-300_438.3_25.11_121.26_-160.0":
                            is_outlier = is_outlier | (module_data_key_not_nan > 3100)
                    elif key=="Irradiance_m":
                        if module=="AUO PM060MW3 320W_246.4_24.107_120.44_4.63":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 260000)
                            else:        is_outlier = is_outlier | (module_data_key_not_nan > 360000)
                        elif module=="AUO PM060MW3 320W_267.52_24.08_120.52_-2.13":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 255000)
                        elif module=="AUO PM060MW3 320W_278.4_24.09_120.52_0.0":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 260000)
                        elif module=="AUO PM060MW3 320W_314.88_24.06_120.47_0.0":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 130000)
                        elif module=="AUO PM060MW3 320W_352.0_24.07_120.47_0.0":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 140000)
                        elif module=="AUO PM060MW3 320W_492.8_24.107_120.44_4.63":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 132500)
                            else:        is_outlier = is_outlier | (module_data_key_not_nan > 180000)
                        elif module=="AUO PM060MW3 320W_498.56_24.04_120.52_2.21":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 131000)
                            else:        is_outlier = is_outlier | (module_data_key_not_nan > 180000)
                        elif module=="AUO PM060MW3 320W_498.56_24.04_120.52_2.21_temp=60":
                            if fit_test: is_outlier = is_outlier
                            else:        is_outlier = is_outlier | (module_data_key_not_nan > 180000)
                        elif module=="AUO PM060MW3 320W_99.2_24.08_120.5_1.76":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 530000)
                            else:        is_outlier = is_outlier | (module_data_key_not_nan > 730000)
                        elif module=="AUO PM060MW3 320W_99.84_24.07_120.48_10.35":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 520000)
                            # else:        is_outlier = is_outlier | (module_data_key_not_nan > 0)
                        # elif module=="AUO PM060MW3 325W_343.2_24.08_120.52_-2.62":
                        #     if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 0)
                        #     else:        is_outlier = is_outlier | (module_data_key_not_nan > 0)
                        elif module=="MM60-6RT-300_438.3_25.11_121.26_-160.0":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 5200)
                            # else:        is_outlier = is_outlier | (module_data_key_not_nan > 0)
                        elif module=="MM60-6RT-300_498.6_25.03_121.08_-95.0":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 5700)
                            # else:        is_outlier = is_outlier | (module_data_key_not_nan > 0)
                        elif module=="MM60-6RT-300_499.8_25.11_121.26_22.0":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 5200)
                            # else:        is_outlier = is_outlier | (module_data_key_not_nan > 0)
                        elif module=="SEC-6M-60A-295_283.2_24.98_121.03_-31.0":
                            if fit_test: is_outlier = is_outlier | (module_data_key_not_nan > 5700)
                            # else:        is_outlier = is_outlier | (module_data_key_not_nan > 0)
                    elif key=="I_m/I":
                        if module=="AUO PM060MW3 320W_246.4_24.107_120.44_4.63":
                            if fit_test: is_outlier = module_data_key_not_nan > 18000
                            else:        is_outlier = module_data_key_not_nan > 18000
                        elif module=="AUO PM060MW3 320W_267.52_24.08_120.52_-2.13":
                            if fit_test: is_outlier = module_data_key_not_nan > 17000
                            else:        is_outlier = module_data_key_not_nan > 17000
                        elif module=="AUO PM060MW3 320W_278.4_24.09_120.52_0.0":
                            if fit_test: is_outlier = module_data_key_not_nan > 17000
                            else:        is_outlier = module_data_key_not_nan > 17000
                        # elif module=="AUO PM060MW3 320W_314.88_24.06_120.47_0.0":
                        #     if fit_test: is_outlier = module_data_key_not_nan > 0
                        #     else:        is_outlier = module_data_key_not_nan > 0
                        # elif module=="AUO PM060MW3 320W_352.0_24.07_120.47_0.0":
                        #     if fit_test: is_outlier = module_data_key_not_nan > 0
                        #     else:        is_outlier = module_data_key_not_nan > 0
                        elif module=="AUO PM060MW3 320W_492.8_24.107_120.44_4.63":
                            if fit_test: is_outlier = module_data_key_not_nan > 9000
                            else:        is_outlier = module_data_key_not_nan > 9000
                        elif module=="AUO PM060MW3 320W_498.56_24.04_120.52_2.21":
                            if fit_test: is_outlier = module_data_key_not_nan > 9000
                            else:        is_outlier = module_data_key_not_nan > 9000
                        # elif module=="AUO PM060MW3 320W_498.56_24.04_120.52_2.21_temp=60":
                        #     if fit_test: is_outlier = module_data_key_not_nan > 180000
                        #     else:        is_outlier = module_data_key_not_nan > 180000
                        elif module=="AUO PM060MW3 320W_99.2_24.08_120.5_1.76":
                            if fit_test: is_outlier = module_data_key_not_nan > 40000
                            else:        is_outlier = module_data_key_not_nan > 40000
                        elif module=="AUO PM060MW3 320W_99.84_24.07_120.48_10.35":
                            if fit_test: is_outlier = module_data_key_not_nan > 35000
                            else:        is_outlier = module_data_key_not_nan > 35000
                        # elif module=="AUO PM060MW3 325W_343.2_24.08_120.52_-2.62":
                        #     if fit_test: is_outlier = module_data_key_not_nan > 0
                        #     else:        is_outlier = module_data_key_not_nan > 0
                        elif module=="MM60-6RT-300_438.3_25.11_121.26_-160.0":
                            if fit_test: is_outlier = module_data_key_not_nan < 276.92 + 0.1
                            else:        is_outlier = module_data_key_not_nan < 276.92 + 0.1
                        # elif module=="MM60-6RT-300_498.6_25.03_121.08_-95.0":
                        #     if fit_test: is_outlier = module_data_key_not_nan < 0
                        #     else:        is_outlier = module_data_key_not_nan < 0
                        elif module=="MM60-6RT-300_499.8_25.11_121.26_22.0":
                            is_outlier = (module_data_key_not_nan > 276.92 + 1.1) | (module_data_key_not_nan < 276.92 + 0.6)
                        # elif module=="SEC-6M-60A-295_283.2_24.98_121.03_-31.0":
                        #     if fit_test: is_outlier = module_data_key_not_nan > 0
                        #     else:        is_outlier = module_data_key_not_nan > 0
                    elif key == "I/I_m":
                        if module=="AUO PM060MW3 320W_246.4_24.107_120.44_4.63":
                            is_outlier = module_data_key_not_nan > 2
                        elif module=="AUO PM060MW3 320W_267.52_24.08_120.52_-2.13":
                            is_outlier = module_data_key_not_nan > 0.001
                        elif module=="AUO PM060MW3 320W_278.4_24.09_120.52_0.0":
                            is_outlier = module_data_key_not_nan > 0.001
                        elif module=="AUO PM060MW3 320W_314.88_24.06_120.47_0.0":
                            is_outlier = module_data_key_not_nan > 0.002
                        elif module=="AUO PM060MW3 320W_352.0_24.07_120.47_0.0":
                            is_outlier = module_data_key_not_nan > 0.002
                        elif module=="AUO PM060MW3 320W_492.8_24.107_120.44_4.63":
                            is_outlier = module_data_key_not_nan > 0.002
                        elif module=="AUO PM060MW3 320W_498.56_24.04_120.52_2.21":
                            is_outlier = module_data_key_not_nan > 4
                        # elif module=="AUO PM060MW3 320W_498.56_24.04_120.52_2.21_temp=60":
                        #     if fit_test: is_outlier = module_data_key_not_nan > 0
                        #     else:        is_outlier = module_data_key_not_nan > 0
                        elif module=="AUO PM060MW3 320W_99.2_24.08_120.5_1.76":
                            is_outlier = module_data_key_not_nan > 0.001
                        elif module=="AUO PM060MW3 320W_99.84_24.07_120.48_10.35":
                            is_outlier = module_data_key_not_nan > 0.001
                        # elif module=="AUO PM060MW3 325W_343.2_24.08_120.52_-2.62":
                        #     if fit_test: is_outlier = module_data_key_not_nan > 0
                        #     else:        is_outlier = module_data_key_not_nan > 0
                        elif module=="MM60-6RT-300_438.3_25.11_121.26_-160.0":
                            is_outlier = (module_data_key_not_nan > 0.003595 + 8e-6) | (module_data_key_not_nan < 0.003595 + 1e-6)
                        # elif module=="MM60-6RT-300_498.6_25.03_121.08_-95.0":
                        #     if fit_test: is_outlier = module_data_key_not_nan > 0
                        #     else:        is_outlier = module_data_key_not_nan > 0
                        elif module=="MM60-6RT-300_499.8_25.11_121.26_22.0":
                            if fit_test: is_outlier = (module_data_key_not_nan > 0.003595 + 8e-6) | (module_data_key_not_nan < 0.003595 + 1e-6)
                            else:        is_outlier = (module_data_key_not_nan > 0.003595 + 1e-5) | (module_data_key_not_nan < 0.003595 + 1e-6)
                        # elif module=="SEC-6M-60A-295_283.2_24.98_121.03_-31.0":
                        #     if fit_test: is_outlier = module_data_key_not_nan > 0
                        #     else:        is_outlier = module_data_key_not_nan > 0

                    module_data_key_outlier = module_data_key_not_nan[is_outlier]
                    module_data_key_not_nan = module_data_key_not_nan[~is_outlier]
                    module_data_key_outlier = pd.Series([np.nan]*len(module_data_key_outlier), index=module_data_key_outlier.index)
                    module_data_key = pd.concat([module_data_key_nan, module_data_key_outlier, module_data_key_not_nan])
                    module_data_key.sort_index(inplace=True)
                    module_data[module][key] = module_data_key

                # Do normalization only on values which are not NaN
                module_data_key         = module_data[module][key]
                module_data_key_nan     = module_data_key[module_data_key.isna()]
                module_data_key_not_nan = module_data_key[module_data_key.notna()]

                if normalize_settings[key] == "range":
                    module_data_key_not_nan -= module_data_key_not_nan.min()
                    if module_data_key_not_nan.max() != 0:
                        module_data_key_not_nan /= module_data_key_not_nan.max()
                elif normalize_settings[key] == "max_abs":
                    raise NotImplementedError
                    module_data_key_not_nan += module_data_key_not_nan.abs().max()
                    if module_data_key_not_nan.max() != 0:
                        module_data_key_not_nan /= module_data_key_not_nan.max()
                elif normalize_settings[key] == "original" and key != "Generation":
                    # module_data_key_not_nan -= module_data_key_not_nan.min()
                    pass

                if key == "Temp_m":
                    # module_data_key_nan.fillna(-1.0, inplace=True)
                    module_data_key_nan.fillna(0.0, inplace=True)

                module_data_key = pd.concat([module_data_key_nan, module_data_key_not_nan])
                module_data_key.sort_index(inplace=True)
                module_data[module][key] = module_data_key

                if key == "Irradiance_m":
                    module_data[module]["I_m/I-min"] = (module_data[module]["Irradiance_m"] + 1e-9)/(module_data[module]["Irradiance"] + 1e-9)
                    module_data[module]["I/I_m-min"] = (module_data[module]["Irradiance"] + 1e-9)/(module_data[module]["Irradiance_m"] + 1e-9)

        data = pd.concat(module_data.values())
        data.sort_index(inplace=True)

    else:
        for key in standard_normalize_keys:

            # if normalize_settings["Outlier"] in ["NaN", "remove"]:
            #     is_outlier = (data[key]-data[key].mean()).abs() <= 3*data[key].std()
            #     data = data[is_outlier]

            data_key         = data[key]
            data_key_nan     = data_key[data_key.isna()]
            data_key_not_nan = data_key[data_key.notna()]

            assert normalize_settings[key] in ["range", "max_abs", "original"]
            if normalize_settings[key] == "range":
                data_key_not_nan -= data_key_not_nan.min()
                data_key_not_nan /= data_key_not_nan.max()
            elif normalize_settings[key] == "max_abs":
                raise NotImplementedError
                data_key_not_nan += data_key_not_nan.abs().max()
                data_key_not_nan /= data_key_not_nan.max()
            # elif normalize_settings[key] == "original" and key != "Generation":
            #     data_key_not_nan -= data_key_not_nan.min()

            data_key = pd.concat([data_key_nan, data_key_not_nan])
            data_key.sort_index(inplace=True)
            data[key] = data_key


    # Process test_data
    test_data = data[-len(test_data):]
    test_data["Generation"].fillna(0.0, inplace=True)

    # Process train_valid_data
    train_valid_data = data[:len(train_data)]
    train_valid_data = train_valid_data[train_valid_data["Temp"].notna()]
    train_valid_data = train_valid_data[train_valid_data["Irradiance"].notna()]
    train_valid_data = train_valid_data[train_valid_data["Irradiance_m"].notna()]
    train_valid_data = train_valid_data[train_valid_data["I_m/I"].notna()]
    train_valid_data = train_valid_data[train_valid_data["I/I_m"].notna()]
    train_valid_data = train_valid_data[train_valid_data["Generation"].notna()]

    # Split train_valid_data into train_data & valid_data
    train_data, valid_data = [], []
    for newer_module in MODULES:
        module_data = train_valid_data[train_valid_data[normalize_settings["Mode"]]==newer_module]
        module_train_data = module_data.sample(frac=0.8, random_state=0)  # random state is a seed value
        module_valid_data = module_data.drop(module_train_data.index)
        train_data.append(module_train_data)
        valid_data.append(module_valid_data)
    train_data = pd.concat(train_data)
    valid_data = pd.concat(valid_data)

    if normalize_settings["Module"] == "one-hot":
        for newer_module in MODULES:
            train_data[newer_module] = pd.Series(train_data[normalize_settings["Mode"]]==newer_module, dtype=int)
            valid_data[newer_module] = pd.Series(valid_data[normalize_settings["Mode"]]==newer_module, dtype=int)
            test_data[newer_module]  = pd.Series(test_data[normalize_settings["Mode"]]==newer_module, dtype=int)
        train_data.drop(normalize_settings["Mode"], inplace=True, axis=1)
        valid_data.drop(normalize_settings["Mode"], inplace=True, axis=1)
        test_data.drop( normalize_settings["Mode"], inplace=True, axis=1)


    if part=="All":
        return pd.concat([train_data, valid_data, test_data])
    elif part=="Train+Valid":
        return pd.concat([train_data, valid_data])
    elif part=="Train":
        return train_data
    elif part=="Valid":
        return valid_data
    elif part=="Test":
        return test_data
    else:
        raise Exception


def plot_auto_training(history, keys, LCB_coefficient, LCB_exponent,
                       turn, best_turn, cost_time, filename, save_dir):
    fig, axs = plt.subplots(1, len(keys), figsize=(len(keys)*3, 10))
    second = cost_time % 60
    minute = cost_time // 60 % 60
    hour   = cost_time // 60 // 60
    title_string = f"Results\n(search turn - {turn} / cost time - {hour}:{minute}:{second})"
    if len(history) != 0:
        best_hyperparameter = dict([ (key, history[best_turn][key]) for key in keys ])
        title_string += f"\n(best turn: {best_turn+1:04} - hyperparameter: {best_hyperparameter})"
    fig.suptitle(title_string)
    
    for ki, key in enumerate(keys):

        x_loss_ids = [ h[key] for h in history ]
        train_loss = [ h["train_loss"] for h in history ]
        valid_loss = [ h["valid_loss"] for h in history ]

        x_mean_ids = list(set(x_loss_ids))
        valid_mean = { x: [] for x in x_mean_ids }
        for vli, vl in enumerate(valid_loss):
            valid_mean[x_loss_ids[vli]].append(vl)
        valid_delta = [ math.sqrt(LCB_coefficient*(math.log(len(history))/len(vl)**LCB_exponent)) for vl in list(valid_mean.values()) ]
        valid_mean = [ sum(vl)/len(vl) for vl in list(valid_mean.values()) ]

        axs[ki].errorbar(x_mean_ids, valid_mean, valid_delta, fmt='o', ms=10, linewidth=5,
                         capsize=8, capthick=3, c="orange", label="valid confidence", zorder=1)
        axs[ki].scatter(x_loss_ids, train_loss, s=10, c="blue", label="train loss", zorder=2)
        axs[ki].scatter(x_loss_ids, valid_loss, s=10, c="red" , label="valid loss", zorder=2)
        axs[ki].set_title(key)
        axs[ki].legend()
        axs[ki].grid()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}.png")
    plt.close("all")
    return