""" Libraries """
import torch
import numpy as np
from functions_new import get_normalized_data



""" Functions """
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, part, drop_keys=["Generation"], fit_test=False, module=None):
        super(TrainingDataset).__init__()
        assert part in ["Train", "Valid"]

        if "Generation" not in drop_keys:
            drop_keys.append("Generation")

        module_setting = "one-hot"
        # if module is not None:
        #     module_setting = "original"

        data = get_normalized_data(part, module_setting, fit_test)

        self.truth = data["Generation"].values
        # if module != None:
        #     data = data[data["Module"]==module]
        #     data.drop("Module", axis=1, inplace=True)

        for drop_key in drop_keys:
            data.drop(drop_key, axis=1, inplace=True)
        self.input = data.values

    def __getitem__(self, index):
        truth = self.truth[index]
        input = self.input[index]
        return np.array(input, dtype=np.float32), np.array(truth, dtype=np.float32)

    def __len__(self):
        return len(self.input)


class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, drop_keys=["Generation"]):
        super(TrainingDataset).__init__()
        data = get_normalized_data(part="Test", module_setting="one-hot", fit_test=False)
        if "Generation" not in drop_keys:
            drop_keys.append("Generation")
        for drop_key in drop_keys:
            data.drop(drop_key, axis=1, inplace=True)
        self.input = data.values

    def __getitem__(self, index):
        input = self.input[index]
        return np.array(input, dtype=np.float32)

    def __len__(self):
        return len(self.input)