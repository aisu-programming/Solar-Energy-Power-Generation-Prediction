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


def get_normalized_data(part="Train", module_setting="one-hot"):

    assert module_setting in ["one-hot", "original"]
    assert part           in ["All", "Train+Valid", "Train", "Valid", "Test"]

    train_data = pd.read_csv("data/train_v4.csv")
    test_data  = pd.read_csv("data/test_v4_filled.csv")
    data       = pd.concat((train_data, test_data))
    data.reset_index(inplace=True)
    data.drop("ID", inplace=True, axis=1)
    data.drop("index", inplace=True, axis=1)

    datetime = pd.to_datetime(data["Date"])
    year  = (datetime.dt.year.values-2020) / 2
    month = (datetime.dt.month.values-1) / 11
    day   = (datetime.dt.day.values-1) / 30
    data["sin(Year)"]  = pd.Series((np.sin(year*(2*math.pi))+1)/2)
    data["cos(Year)"]  = pd.Series((np.cos(year*(2*math.pi))+1)/2)
    data["sin(Month)"] = pd.Series((np.sin(month*(2*math.pi))+1)/2)
    data["cos(Month)"] = pd.Series((np.cos(month*(2*math.pi))+1)/2)
    data["sin(Day)"]   = pd.Series((np.sin(day*(2*math.pi))+1)/2)
    data["cos(Day)"]   = pd.Series((np.cos(day*(2*math.pi))+1)/2)
    data.drop("Date", inplace=True, axis=1)

    STANDARD_NORMALIZE_KEYS = ["Generation", "Temp", "Temp_m", "Irradiance", "Irradiance_m", 
                               "Capacity", "Lat", "Lon", "Angle",
                               "測站氣壓", "海平面氣壓", "測站最高氣壓", "測站最低氣壓",
                               "氣溫", "最高氣溫", "最低氣溫", "露點溫度", "相對溼度", "最小相對溼度",
                               "風速", "風向", "最大陣風", "最大陣風風向",
                               "降水量", "降水時數", "日照時數", "日照率", "全天空日射量", "能見度",
                               "日最高紫外線指數", "總雲量", "UV" ]
    
    RAW_MODULES = [ "MM60-6RT-300", "SEC-6M-60A-295", "AUO PM060MW3 320W", "AUO PM060MW3 325W" ]
    PMAXS       = [          300.0,            295.0,               320.0,               325.0 ]
    VMPS        = [          32.61,             31.6,               33.48,               33.66 ]
    IMPS        = [            9.2,             9.34,                9.56,                9.66 ]
    VOCS        = [          38.97,             39.4,                40.9,                41.1 ]
    ISCS        = [           9.68,             9.85,               10.24,               10.35 ]
    EFFICACYS   = [          18.44,            17.74,                19.2,                19.5 ]
    NEW_MODULES = [ "MM60-6RT-300", "SEC-6M-60A-295", "AUO PM060MW3" ]

    MODULE_EXTRA_KEYS = ["Pmax", "Vmp", "Imp", "Voc", "Isc", "Efficacy"]
    for key in MODULE_EXTRA_KEYS:
        data[key] = 0.0
    for module, pmax, vmp, imp, voc, isc, eff in zip(RAW_MODULES, PMAXS, VMPS, IMPS, VOCS, ISCS, EFFICACYS):
        for key, value in zip(MODULE_EXTRA_KEYS, [pmax, vmp, imp, voc, isc, eff]):
            data[key][data["Module"]==module] = value
    data["Module"][data["Module"]=="AUO PM060MW3 320W"] = "AUO PM060MW3"
    data["Module"][data["Module"]=="AUO PM060MW3 325W"] = "AUO PM060MW3"

    # for module in NEW_MODULES:
    #     print(data[data["Module"]==module][:2])

    module_data = {}
    for module in NEW_MODULES:
        module_data[module] = data[data["Module"]==module]

        for key in STANDARD_NORMALIZE_KEYS:

            module_data_key         = module_data[module][key]
            module_data_key_nan     = module_data_key[module_data_key.isna()]
            module_data_key_not_nan = module_data_key[module_data_key.notna()]

            # Process outliers
            if key in ["Generation", "Temp_m", "Irradiance_m", "降水量"]:
                
                if   key == "Generation"  : olr = outlier_rate = 3
                elif key == "Temp_m"      : olr = outlier_rate = 3
                elif key == "Irradiance_m": olr = outlier_rate = 3
                elif key == "降水量"       : olr = outlier_rate = 3.5
                is_outlier = (module_data_key_not_nan-module_data_key_not_nan.mean()).abs() > olr*module_data_key_not_nan.std()

                module_data_key_outlier = module_data_key_not_nan[is_outlier]
                module_data_key_not_nan = module_data_key_not_nan[~is_outlier]
                module_data_key_outlier = pd.Series([np.nan]*len(module_data_key_outlier), index=module_data_key_outlier.index)
                module_data_key = pd.concat([module_data_key_nan, module_data_key_outlier, module_data_key_not_nan])
                module_data_key.sort_index(inplace=True)
                module_data[module][key] = module_data_key

            if key != "Generation":
                # Do normalization only on values which are not NaN
                module_data_key         = module_data[module][key]
                module_data_key_nan     = module_data_key[module_data_key.isna()]
                module_data_key_not_nan = module_data_key[module_data_key.notna()]

                module_data_key_not_nan -= module_data_key_not_nan.min()
                if module_data_key_not_nan.max() != 0:
                    module_data_key_not_nan /= module_data_key_not_nan.max()

                if key == "Temp_m":
                    module_data_key_nan.fillna(0.0, inplace=True)

                module_data_key = pd.concat([module_data_key_nan, module_data_key_not_nan])
                module_data_key.sort_index(inplace=True)
                module_data[module][key] = module_data_key

    data = pd.concat(module_data.values())
    data.sort_index(inplace=True)

    for key in MODULE_EXTRA_KEYS:
        data[key] -= data[key].min()
        if data[key].max() != 0:
            data[key] /= data[key].max()

    # Process test_data
    test_data = data[-len(test_data):]
    test_data["Generation"].fillna(0.0, inplace=True)
    assert len(test_data) == test_data["Generation"].notna().sum()
    assert len(test_data) == test_data["Temp"].notna().sum()
    assert len(test_data) == test_data["Temp_m"].notna().sum()
    assert len(test_data) == test_data["Irradiance"].notna().sum()
    assert len(test_data) == test_data["Irradiance_m"].notna().sum()
    assert len(test_data) == test_data["降水量"].notna().sum()

    # Process train_valid_data
    train_valid_data = data[:len(train_data)]
    train_valid_data = train_valid_data[train_valid_data["Generation"].notna()]
    train_valid_data = train_valid_data[train_valid_data["Temp"].notna()]
    train_valid_data = train_valid_data[train_valid_data["Irradiance"].notna()]
    train_valid_data = train_valid_data[train_valid_data["Irradiance_m"].notna()]
    train_valid_data = train_valid_data[train_valid_data["最低氣溫"].notna()]
    train_valid_data = train_valid_data[train_valid_data["最小相對溼度"].notna()]
    train_valid_data = train_valid_data[train_valid_data["降水量"].notna()]
    train_valid_data = train_valid_data[train_valid_data["日最高紫外線指數"].notna()]
    train_valid_data = train_valid_data[train_valid_data["UV"].notna()]

    # Split train_valid_data into train_data & valid_data
    train_data, valid_data = [], []
    for module in NEW_MODULES:
        module_data = train_valid_data[train_valid_data["Module"]==module]
        module_train_data = module_data.sample(frac=0.8, random_state=0)  # random state is a seed value
        module_valid_data = module_data.drop(module_train_data.index)
        train_data.append(module_train_data)
        valid_data.append(module_valid_data)
    train_data = pd.concat(train_data)
    valid_data = pd.concat(valid_data)

    if module_setting == "one-hot":
        for module in NEW_MODULES:
            train_data[module] = pd.Series(train_data["Module"]==module, dtype=int)
            valid_data[module] = pd.Series(valid_data["Module"]==module, dtype=int)
            test_data[module]  = pd.Series(test_data["Module"]==module, dtype=int)
        train_data.drop("Module", inplace=True, axis=1)
        valid_data.drop("Module", inplace=True, axis=1)
        test_data.drop("Module", inplace=True, axis=1)

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