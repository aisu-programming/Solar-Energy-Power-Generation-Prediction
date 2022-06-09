""" Libraries """
import os
import math
import time
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

from model import MyModel
from dataset import TrainingDataset
from functions import seed_worker, get_lr, plot_auto_training

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
TORCH_GENERATOR = torch.Generator()
TORCH_GENERATOR.manual_seed(SEED)



""" CONTANTS """
SEARCH_TRAINING_OPTIONS = {
    "batch_size": [4, 8, 16],
    "dropout"   : [0.0, 0.2, 0.4],
    "lr"        : [0.03, 0.02, 0.01],
    "lradj"     : [0.9997, 0.9995, 0.9993],
    # "depth"     : [3, 4, 5, 6, 7],
    # "width"     : [1.5, 2, 3, 4],
}
LCB_COEFFICIENT = 2000
LCB_EXPONENT    = 2.5
SEARCH_RANDOM_P = 0.2
EPOCH = 100
PATIENCE = 8
# BATCH_SIZE = 32
# DROPOUT = 0.8
# LR = 0.006
# REPLAY = 50
REPLAY_DIVISOR = 4
assert REPLAY_DIVISOR <= min(SEARCH_TRAINING_OPTIONS["batch_size"])
DROP_KEYS = ["Generation", "Date", "Capacity", "Lat", "Lon", "Angle"]
# DROP_KEYS = ["Generation", "Temp"]
# DROP_KEYS = ["Generation", "Temp", "Temp_m"]
SAVE_ROOTDIR = "records_auto_search"
NS = {
    "Date"        : "original",
    "Generation"  : "original",
    # "Temp"        : "original",
    # "Temp_m"      : "original",
    # "Irradiance"  : "original",
    # "Irradiance_m": "original",
}



""" Functions """
def my_setattr(args, key, value):
    # if key == "batch_size":
    #     args.batch_size = 2**value
    # elif key == "learning_rate":
    #     args.learning_rate = 10**(-value)
    # else:
    setattr(args, key, value)
    return args


def LCB_select(history, key):
    options = { v: [
        h["valid_loss"] for h in filter(lambda h: h[key]==v, history)
    ] for v in SEARCH_TRAINING_OPTIONS[key] }
    for v in options.keys():
        if options[v] == []:
            return v
        else:
            if random.random() <= SEARCH_RANDOM_P:
                return random.choice(list(options.keys()))
            mean  = sum(options[v]) / len(options[v])
            delta = math.sqrt(LCB_COEFFICIENT*(math.log(len(history))/len(options[v])**LCB_EXPONENT))
            options[v] = mean - delta
    options = dict(sorted(options.items(), key=lambda o: o[1]))
    return list(options.keys())[0]


def train_epoch(device, model, dataloader, loss_fn, optimizer, lr_scheduler, scaler, epoch):

    losses = []
    nan_num = 0

    pbar = tqdm(dataloader, total=len(dataloader), ascii=True)
    for inputs, truths in pbar:

        inputs, truths = inputs.to(device), truths.to(device)
        with amp.autocast():
            output = model(inputs).squeeze()
            # print(output.shape, truths.shape)
            loss = loss_fn(output, truths)**0.5

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()
        # lr_scheduler.step(loss.item())

        if not torch.isnan(loss):
            losses.append(loss.item())
            avg_loss = np.average(losses)
        else:
            nan_num += 1
            if len(losses)==0: avg_loss = torch.nan

        pbar.set_description(f"Epoch: {epoch} [Train] Avg RMSE: {avg_loss:.2f}, LR: {get_lr(optimizer):.10f}, Nan: {nan_num}")
        # pbar.set_description(f"Epoch: {epoch} [Train] Avg Loss: {avg_loss:.6f}, Avg RMSE: {(avg_loss**0.5)*3263:.2f}, LR: {get_lr(optimizer):.10f}, Nan: {nan_num}")
        
    return avg_loss, nan_num
    # return np.average(losses), nan_num


def valid_epoch(device, model, dataloader, loss_fn, epoch):

    losses = []
    nan_num = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), ascii=True)
        for inputs, truths in pbar:

            inputs, truths = inputs.to(device), truths.to(device)
            with amp.autocast():
                output = model(inputs).squeeze()
                loss = loss_fn(output, truths)**0.5

            if not torch.isnan(loss):
                losses.append(loss.item())
                avg_loss = np.average(losses)
            else:
                nan_num += 1
                if len(losses)==0: avg_loss = torch.nan

            pbar.set_description(f"Epoch: {epoch} [Valid] Avg RMSE: {avg_loss:.2f}, Nan: {nan_num}")
            # pbar.set_description(f"Epoch: {epoch} [Valid] Avg Loss: {avg_loss:.6f}, Avg RMSE: {(avg_loss**0.5)*3263:.2f}, Nan: {nan_num}")
        
    return avg_loss, nan_num
    # return np.average(losses), nan_num


def main(turn, device, args):

    args.replay = int(args.batch_size/REPLAY_DIVISOR)
    train_dataset = TrainingDataset("Train", args.replay, args.drop_keys, NS)
    valid_dataset = TrainingDataset("Valid", args.replay, args.drop_keys, NS)
    
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=TORCH_GENERATOR
    )
    valid_dataloader = DataLoader(
        valid_dataset, args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=TORCH_GENERATOR
    )

    input_dim = 24 - len(args.drop_keys)
    model = MyModel(args.dropout, 14, input_dim-14).to(device)
    # if resume: model.load_state_dict(torch.load(resume_model_path))

    shutil.copy("train.py", args.save_dir)
    shutil.copy("model.py", args.save_dir)
    shutil.copy("dataset.py", args.save_dir)
    shutil.copy("functions.py", args.save_dir)
    tb_writer = SummaryWriter(log_dir=args.save_dir)

    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.998, patience=12, threshold=1e-5, eps=0)
    scaler = amp.GradScaler()

    best_train_avg_rmse, best_valid_avg_rmse, patience = 10000, 10000, 0
    for epoch in range(1, args.epoch+1):
        print('')
        train_avg_rmse, train_nan_num = train_epoch(device, model, train_dataloader, loss_fn, optimizer, lr_scheduler, scaler, epoch)
        valid_avg_rmse, valid_nan_num = valid_epoch(device, model, valid_dataloader, loss_fn, epoch)
        tb_writer.add_scalar(f"Train Avg Loss (Auto)/{turn:04}", train_avg_rmse, epoch)
        tb_writer.add_scalar(f"Train NaN number (Auto)/{turn:04}", train_nan_num, epoch)
        tb_writer.add_scalar(f"Valid Avg Loss (Auto)/{turn:04}", valid_avg_rmse, epoch)
        tb_writer.add_scalar(f"Valid NaN number (Auto)/{turn:04}", valid_nan_num, epoch)
        tb_writer.add_scalar(f"Learning Rate (Auto)/{turn:04}", get_lr(optimizer), epoch)
        if train_avg_rmse < best_train_avg_rmse:
            best_train_avg_rmse = train_avg_rmse
        if valid_avg_rmse < best_valid_avg_rmse - 1:
            best_valid_avg_rmse = valid_avg_rmse
            model_name = f"{args.save_dir}/{turn:04}_best.pt"
            torch.save(model.state_dict(), model_name)
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Stop because of patience\n")
                break
        print(f"Best average RMSE now: {best_valid_avg_rmse:.2f}")

    return best_train_avg_rmse, best_valid_avg_rmse



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
    # parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="")
    # parser.add_argument("--dropout", type=float, default=DROPOUT, help="")
    # parser.add_argument("--lr", type=float, default=LR, help="")
    # parser.add_argument("--replay", type=int, default=REPLAY, help="")
    parser.add_argument("--drop_keys", type=list, default=DROP_KEYS, help="")

    args = parser.parse_args()
    # extra_desc  = f"r={args.replay}"
    # extra_desc  = f"bs={args.batch_size}_d={args.dropout}_lr={args.lr}_r={args.replay}"
    save_subdir = datetime.now().strftime(f"%m.%d-%H.%M.%S")
    # save_subdir = datetime.now().strftime(f"%m.%d-%H.%M.%S_{extra_desc}")
    save_dir    = f"{SAVE_ROOTDIR}/{save_subdir}"
    parser.add_argument("--save_dir", type=str, default=save_dir, help="")
    args = parser.parse_args()
    print('\nStart Tensorboard with "tensorboard --logdir records", view at http://localhost:6006/\n')


    print("Start searching hyperparameters!\n")
    history = []
    best_valid_loss, best_turn = math.inf, 0
    keys = list(SEARCH_TRAINING_OPTIONS.keys())
    start_time = time.time()
    for turn in range(1, 1000+1):

        combination = {}
        for key, value in SEARCH_TRAINING_OPTIONS.items():
            value = LCB_select(history, key)
            combination[key] = value
            args = my_setattr(args, key, value)

        print("Start train with hyperparameters:", combination)
        os.makedirs(args.save_dir, exist_ok=True)
        with open(f"{args.save_dir}/records.txt", 'a') as file:
            file.write(f"{turn:04} - {combination}")
        train_loss, valid_loss = main(turn, DEVICE, args)

        combination["train_loss"] = train_loss
        combination["valid_loss"] = valid_loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_turn = turn-1
        with open(f"{args.save_dir}/records.txt", 'a') as file:
            file.write(f" - train_loss: {train_loss}, valid_loss: {valid_loss}\n")
        history.append(combination)
        plot_auto_training(
            history, keys, LCB_COEFFICIENT, LCB_EXPONENT, turn, best_turn, 
            int(time.time()-start_time), "results", args.save_dir,
        )