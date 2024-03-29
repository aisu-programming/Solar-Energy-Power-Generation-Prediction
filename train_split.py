""" Libraries """
import os
import torch
import shutil
import random
import argparse
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import MySequential
from dataset import TrainingDataset
from functions import seed_worker, get_lr

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
TORCH_GENERATOR = torch.Generator()
TORCH_GENERATOR.manual_seed(SEED)



""" CONTANTS """
EPOCH = 500
PATIENCE = 50
BATCH_SIZE = 32
# DROPOUT = 1.0
LR = 0.25
LRADJ = 0.997
MODULE = "AUO PM060MW3 320W_246.4_24.107_120.44_4.63"
# NEWER_MODULES = ["AUO PM060MW3 320W_246.4_24.107_120.44_4.63", "AUO PM060MW3 320W_267.52_24.08_120.52_-2.13",
#                  "AUO PM060MW3 320W_278.4_24.09_120.52_0.0", "AUO PM060MW3 320W_314.88_24.06_120.47_0.0",
#                  "AUO PM060MW3 320W_352.0_24.07_120.47_0.0", "AUO PM060MW3 320W_492.8_24.107_120.44_4.63",
#                  "AUO PM060MW3 320W_498.56_24.04_120.52_2.21", "AUO PM060MW3 320W_498.56_24.04_120.52_2.21_temp=60",
#                  "AUO PM060MW3 320W_99.2_24.08_120.5_1.76", "AUO PM060MW3 320W_99.84_24.07_120.48_10.35",
#                  "AUO PM060MW3 325W_343.2_24.08_120.52_-2.62", "MM60-6RT-300_438.3_25.11_121.26_-160.0",
#                  "MM60-6RT-300_498.6_25.03_121.08_-95.0", "MM60-6RT-300_499.8_25.11_121.26_22.0",
#                  "SEC-6M-60A-295_283.2_24.98_121.03_-31.0"]
DROP_KEYS = ["Generation", "Date", "Capacity", "Lat", "Lon", "Angle",
             "I_m/I", "I/I_m", "I_m/I-min", "I/I_m-min"]
# DROP_KEYS = ["Generation", "Temp", "Temp_m", "Date", "Capacity", "Lat", "Lon", "Angle",
#              "I_m/I", "I/I_m", "I_m/I-min", "I/I_m-min"]
FIT_TEST = False
SAVE_ROOTDIR = f"records/{MODULE}"
NS = {
    "Date"        : "original",
    # "Module"      : "original",
    "Generation"  : "original",
    # "Temp"        : "original",
    # "Temp_m"      : "original",
    # "Irradiance"  : "original",
    # "Irradiance_m": "original",
    # "I_m/I"       : "original",
    # "I/I_m"       : "original",
    # "I_m/I-min"   : "original",
    # "I/I_m-min"   : "original",
}



""" Functions """
def train_epoch(device, model, dataloader, loss_fn, optimizer, lr_scheduler, scaler, epoch):

    losses = []
    nan_num = 0

    pbar = tqdm(dataloader, total=len(dataloader), ascii=True)
    for inputs, truths in pbar:

        inputs, truths = inputs.to(device), truths.to(device)
        with amp.autocast():
            output = model(inputs).reshape((-1))
            # print(inputs.shape, output.shape, truths.shape)
            if torch.isnan(output).any():
                nan = torch.nonzero(torch.isnan(output)).flatten()
                print("NaN detected:", nan)
                for n in nan:
                    print(inputs[n])
            loss = loss_fn(output, truths)**0.5

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()
        # lr_scheduler.step(loss)

        if not torch.isnan(loss):
            losses.append(loss.item())
            avg_loss = np.average(losses)
        else:
            nan_num += 1
            if len(losses)==0: avg_loss = torch.nan

        pbar.set_description(f"Epoch: {epoch} [Train] Avg RMSE: {avg_loss:.2f}, LR: {get_lr(optimizer):.10f}, Nan: {nan_num}")
        
    return avg_loss, nan_num


def valid_epoch(device, model, dataloader, loss_fn, epoch):

    losses = []
    nan_num = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), ascii=True)
        for inputs, truths in pbar:

            inputs, truths = inputs.to(device), truths.to(device)
            with amp.autocast():
                output = model(inputs).reshape((-1))
                if torch.isnan(output).any():
                    nan = torch.nonzero(torch.isnan(output)).flatten()
                    print("NaN detected:", nan)
                    for n in nan:
                        print(inputs[n])
                loss = loss_fn(output, truths)**0.5

            if not torch.isnan(loss):
                losses.append(loss.item())
                avg_loss = np.average(losses)
            else:
                nan_num += 1
                if len(losses)==0: avg_loss = torch.nan

            pbar.set_description(f"Epoch: {epoch} [Valid] Avg RMSE: {avg_loss:.2f}, Nan: {nan_num}")
        
    return avg_loss, nan_num


def main(device, args):

    train_dataset = TrainingDataset("Train", 1, args.drop_keys, NS, args.fit_test, MODULE)
    valid_dataset = TrainingDataset("Valid", 1, args.drop_keys, NS, args.fit_test, MODULE)
    
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=TORCH_GENERATOR
    )
    valid_dataloader = DataLoader(
        valid_dataset, args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=TORCH_GENERATOR
    )

    input_dim = 14 - len(args.drop_keys)
    model = MySequential(input_dim).to(device)

    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy("train.py", args.save_dir)
    shutil.copy("model.py", args.save_dir)
    shutil.copy("dataset.py", args.save_dir)
    shutil.copy("functions.py", args.save_dir)
    tb_writer = SummaryWriter(log_dir=args.save_dir)

    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lradj)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.975, patience=30, threshold=0.1, eps=0)
    scaler = amp.GradScaler()

    best_valid_avg_rmse = 10000
    for epoch in range(1, args.epoch+1):

        train_avg_rmse, train_nan_num = train_epoch(device, model, train_dataloader, loss_fn, optimizer, lr_scheduler, scaler, epoch)
        valid_avg_rmse, valid_nan_num = valid_epoch(device, model, valid_dataloader, loss_fn, epoch)

        tb_writer.add_scalar(f"Avg Loss/Train", train_avg_rmse, epoch)
        tb_writer.add_scalar(f"NaN number/Train", train_nan_num, epoch)
        tb_writer.add_scalar(f"Avg Loss/Valid", valid_avg_rmse, epoch)
        tb_writer.add_scalar(f"NaN number/Valid", valid_nan_num, epoch)
        tb_writer.add_scalar(f"Learning Rate", get_lr(optimizer), epoch)

        if epoch >= 2:
            if valid_avg_rmse < best_valid_avg_rmse - 0.5:
                best_valid_avg_rmse = valid_avg_rmse
                model_name = f"{args.save_dir}/best.pt"
                torch.save(model.state_dict(), model_name)
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("Stop because of patience\n")
                    break
            print(f"Best average RMSE now: {best_valid_avg_rmse:.2f}")
        print('')

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
    parser.add_argument("--epoch", type=int, default=EPOCH, help="")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="")
    # parser.add_argument("--dropout", type=float, default=DROPOUT, help="")
    parser.add_argument("--lr", type=float, default=LR, help="")
    parser.add_argument("--lradj", type=float, default=LRADJ, help="")
    parser.add_argument("--drop_keys", type=list, default=DROP_KEYS, help="")
    parser.add_argument("--fit_test", type=bool, default=FIT_TEST, help="")

    args = parser.parse_args()
    extra_desc    = f"bs={args.batch_size}_lr={args.lr}_lradj={args.lradj}_ft={args.fit_test}"
    save_subdir   = datetime.now().strftime(f"%m.%d-%H.%M.%S_{extra_desc}")
    save_dir      = f"{SAVE_ROOTDIR}/{save_subdir}"
    args.save_dir = save_dir

    print('\nStart Tensorboard with "tensorboard --logdir records", view at http://localhost:6006/\n')

    main(DEVICE, args)