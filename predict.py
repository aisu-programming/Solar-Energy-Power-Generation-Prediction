""" Libraries """
import os
import torch
import shutil
import random
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from torch.cuda import amp
from torch.utils.data import DataLoader

from model import MyModel
# from records.model import MyModel
from dataset import PredictionDataset
from functions import seed_worker

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
TORCH_GENERATOR = torch.Generator()
TORCH_GENERATOR.manual_seed(SEED)



""" CONSTANTS """
RESUME_MODEL_DIR = "records/06.05-14.21.09_bs=32_lr=0.25_lradj=0.997_ft=False"
DROP_KEYS = [ "Generation" ]



""" Functions """
def predict(device, model, dataloader):
    outputs = []
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), ascii=True)
        for i, inputs in pbar:
            inputs = inputs.to(device)
            with amp.autocast():
                output = model(inputs).reshape((-1))
                if torch.isnan(output).any():
                    nan = torch.nonzero(torch.isnan(output)).flatten()
                    print("NaN detected:", nan)
                    for n in nan:
                        print(i*32+n+1, inputs[n])
            outputs += list(output.detach().cpu().numpy().flatten())
    return outputs


def main(device, args):

    dataset = PredictionDataset(args.drop_keys)
    dataloader = DataLoader(
        dataset, args.batch_size, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=TORCH_GENERATOR
    )

    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy("predict.py", args.save_dir)
    shutil.copy("model.py", args.save_dir)
    shutil.copy("dataset.py", args.save_dir)
    shutil.copy("functions.py", args.save_dir)

    input_dim = 21 - len(args.drop_keys)
    model = MyModel(input_dim, device).to(device)
    model.load_state_dict(torch.load(args.resume_model_path))

    outputs = predict(device, model, dataloader)
    prediction = pd.DataFrame({"ID": list(range(1, 1540)), "Generation": outputs})
    prediction.set_index("ID", inplace=True)
    prediction.to_csv(f"{args.save_dir}/prediction.csv")

    return



""" Execution """
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_keys", type=list, default=DROP_KEYS, help="")

    # BATCH_SIZE = int(setting["batch_size"])
    BATCH_SIZE = int(RESUME_MODEL_DIR.split("_lr=")[0].split("_bs=")[1])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="")
    # DROPOUT = float(RESUME_MODEL_DIR.split("_lr=")[0].split("_d=")[1])
    # parser.add_argument("--dropout", type=int, default=DROPOUT, help="")
    FIT_TEST = bool(RESUME_MODEL_DIR.split("_ft=")[1])
    parser.add_argument("--fit_test", type=bool, default=FIT_TEST, help="")
    RESUME_MODEL_PATH = f"{RESUME_MODEL_DIR}/best.pt"
    parser.add_argument("--resume_model_path", type=str, default=RESUME_MODEL_PATH, help="")

    args = parser.parse_args()
    SAVE_ROOTDIR = "predictions"
    extra_desc   = f"bs={args.batch_size}_ft={args.fit_test}"
    SAVE_SUBDIR  = datetime.now().strftime(f"%m.%d-%H.%M.%S_{extra_desc}")
    SAVE_DIR     = f"{SAVE_ROOTDIR}/{SAVE_SUBDIR}"
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="")
    args = parser.parse_args()
    
    main(DEVICE, args)