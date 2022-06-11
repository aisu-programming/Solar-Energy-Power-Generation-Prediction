""" Libraries """
import torch
import numpy as np
# from performer_pytorch import Performer


activation_function = torch.nn.Mish()


""" Functions """
class MySequential(torch.nn.Module):
    def __init__(self, input_dim, ratio=2):
        super(MySequential, self).__init__()
        assert (2*ratio % 1) == 0
        self.layers_5 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(16*ratio)),
            activation_function,
            torch.nn.Linear(int(16*ratio), int(8*ratio)),
            activation_function,
            torch.nn.Linear(int(8*ratio), int(4*ratio)),
            activation_function,
            torch.nn.Linear(int(4*ratio), int(2*ratio)),
            activation_function,
            torch.nn.Linear(int(2*ratio), 1),
            activation_function,
        )
        self.layers_4 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(8*ratio)),
            activation_function,
            torch.nn.Linear(int(8*ratio), int(4*ratio)),
            activation_function,
            torch.nn.Linear(int(4*ratio), int(2*ratio)),
            activation_function,
            torch.nn.Linear(int(2*ratio), 1),
            activation_function,
        )
        self.layers_3 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(4*ratio)),
            activation_function,
            torch.nn.Linear(int(4*ratio), int(2*ratio)),
            activation_function,
            torch.nn.Linear(int(2*ratio), 1),
            activation_function,
        )
        self.layers_2 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(2*ratio)),
            activation_function,
            torch.nn.Linear(int(2*ratio), 1),
            activation_function,
        )
        self.layers_1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            activation_function,
        )
        self.integrate_layers = torch.nn.Sequential(
            torch.nn.Linear(5, 1),
            activation_function,
        )

    def forward(self, x):
        x5 = self.layers_5(x)
        x4 = self.layers_4(x)
        x3 = self.layers_3(x)
        x2 = self.layers_2(x)
        x1 = self.layers_1(x)
        x = torch.concat([x5, x4, x3, x2, x1], dim=1)
        x = self.integrate_layers(x)
        return x


class MyModel(torch.nn.Module):
    def __init__(self, input_dim, device):
        super(MyModel, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.my_sequentials = torch.nn.ModuleList([
            MySequential(input_dim),
            MySequential(input_dim),
            MySequential(input_dim),
        ])

    def __split__(self, batch):
        orig_i  = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [] ]
        class_x = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [] ]
        for xi, x in enumerate(batch):
            orig_i[torch.argmax(x[-3:]).item()].append(xi)
            class_x[torch.argmax(x[-3:]).item()].append(x[:self.input_dim].cpu().detach().numpy())
        class_x = [ torch.tensor(np.array(cx)).to(self.device) for cx in class_x ]
        return orig_i, class_x

    def forward(self, inputs):
        orig_i, class_x = self.__split__(inputs)
        outputs = [ None ] * inputs.shape[0]
        for class_i, (ois, xs) in enumerate(zip(orig_i, class_x)):
            if len(ois) == 0: continue
            xs = self.my_sequentials[class_i](xs)
            for xi, oi in enumerate(ois): outputs[oi] = xs[xi]
        output = torch.concat(outputs)
        return output