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

from model import MyModel
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
LR = 0.2
LRADJ = 0.998
DROP_KEYS = [
    "Generation", "Temp", "Temp_m",
    "測站氣壓", "海平面氣壓", "測站最高氣壓", "測站最低氣壓",
    "氣溫", "最高氣溫", "最低氣溫",
    "露點溫度", "相對溼度", "最小相對溼度",
    "風速", "風向", "最大陣風", "最大陣風風向",
    "降水量", "降水時數", "日照時數", "日照率", "全天空日射量", "能見度",
    "日最高紫外線指數", "總雲量", "UV"
]
SAVE_ROOTDIR = "records"



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

    train_dataset = TrainingDataset("Train", args.drop_keys)
    valid_dataset = TrainingDataset("Valid", args.drop_keys)
    
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=TORCH_GENERATOR
    )
    valid_dataloader = DataLoader(
        valid_dataset, args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=TORCH_GENERATOR
    )

    input_dim = 44 - len(args.drop_keys)
    model = MyModel(input_dim, device).to(device)
    # if resume: model.load_state_dict(torch.load(resume_model_path))

    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy("train.py", args.save_dir)
    shutil.copy("model.py", args.save_dir)
    shutil.copy("dataset.py", args.save_dir)
    shutil.copy("functions_new.py", args.save_dir)
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

    args = parser.parse_args()
    extra_desc    = f"bs={args.batch_size}_lr={args.lr}_lradj={args.lradj}"
    save_subdir   = datetime.now().strftime(f"%m.%d-%H.%M.%S_{extra_desc}")
    save_dir      = f"{SAVE_ROOTDIR}/{save_subdir}"
    args.save_dir = save_dir

    print('\nStart Tensorboard with "tensorboard --logdir records", view at http://localhost:6006/\n')

    main(DEVICE, args)