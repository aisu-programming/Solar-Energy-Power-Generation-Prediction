""" Libraries """
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from matplotlib import animation
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
RESUME_MODEL_DIR = "records/06.11-18.33.02_bs=32_lr=0.2_lradj=0.997_ft=False"
DROP_KEYS = [ "Generation" ]
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")



""" Functions """
def test():

    resume_model_path = f"{RESUME_MODEL_DIR}/best.pt"
    batch_size = int(RESUME_MODEL_DIR.split("_lr=")[0].split("_bs=")[1])

    dataset = PredictionDataset(DROP_KEYS)
    dataloader = DataLoader(
        dataset, batch_size, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=TORCH_GENERATOR
    )

    input_dim = 43 - len(DROP_KEYS)
    model = MyModel(input_dim, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(resume_model_path))

    outputs = []
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), ascii=True)
        for i, inputs in pbar:
            inputs = inputs.to(DEVICE)
            with amp.autocast():
                output = model(inputs).reshape((-1))
                if torch.isnan(output).any():
                    nan = torch.nonzero(torch.isnan(output)).flatten()
                    print("NaN detected:", nan)
                    for n in nan:
                        print(i*64+n+1, inputs[n])
            outputs += list(output.detach().cpu().numpy().flatten())

    import pandas as pd
    from functions_new import get_normalized_data
    train_data = get_normalized_data(part="Train", module_setting="original")
    valid_data = get_normalized_data(part="Valid", module_setting="original")
    test_data  = get_normalized_data(part="Test" , module_setting="original")
    test_data["Generation"] = pd.Series(outputs, index=test_data.index)

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(20, 10), subplot_kw=dict(projection="3d"))

    MODULES = [ "MM60-6RT-300", "SEC-6M-60A-295", "AUO PM060MW3" ]
    for mi, module in enumerate(MODULES):
        for data, color in [(train_data, "r"), (valid_data, "g"), (test_data, "b")]:
            data = data[data["Module"]==module]
            irm = data["Irradiance_m"]
            irr = data["Irradiance"]
            gen = data["Generation"]
            axs[mi].set_title(module, size=12)
            axs[mi].set_xlabel("Irradiance_m")
            axs[mi].set_ylabel("Irradiance")
            axs[mi].set_zlabel("Generation")
            axs[mi].scatter(irm, irr, gen, s=2, color=color)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95)

    # def rotate(i):
    #     for mi in range(len(MODULES)):
    #         axs[mi].view_init(25+20*math.sin((i/90)*math.pi), (i*2)+45+90)
    #     return fig
    # print("Creating animation...")
    # anim = animation.FuncAnimation(fig, rotate, frames=360, interval=100, blit=False)
    # writer = animation.PillowWriter(fps=30)
    # print("Saving animation...")
    # anim.save("test.gif", writer=writer)

    plt.show()
    return



if __name__ == "__main__":
    test()