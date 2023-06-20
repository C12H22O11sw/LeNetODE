import torch.nn
from torch import nn
import torch.nn.functional as F
import torchdiffeq


class ODE_Block(nn.Module):
    """
    Implemetaion of an ODE Block with abilities for back propogation.
    Differentail function is taken as an argument in the constructor.
    
    Requires: Differential function be of type nn.Module,
              Differential function have arguments (self, t:float, x:tensor),
              Differential function have equivilent input and output dimentions.
    """
    
    
    def __init__(self, diff_function: nn.Module, method:str='euler'):
                     
        super().__init__()
        self.diff_function = diff_function
        self.ts = torch.tensor([0.0,1.0])
        self.method = method
  
    def forward(self, x):
        h = torchdiffeq.odeint_adjoint(self.diff_function, x, 
                                       self.ts, 
                                       method=self.method,
                                       atol=1e-1, rtol=1e-2,
                                       adjoint_atol=1e-1, adjoint_rtol=1e-2,
                                       adjoint_options=dict(norm="seminorm")
                                       )
        return h[-1, :, :]



class Mini_ODE_Net(nn.Module):
    """
    Example of a small neural net that uses an ode block.

    Input Image
    ODE_BLock
        Conv2d
        ReLU
    Fully Connected layer
    Output label
    
    """
    class DiffFunction(nn.Module):

        def __init__(self):
            super().__init__()
            dim = 10
            self.net = nn.Sequential(
                nn.Conv2d(1, 1, 3, padding="same"),
                nn.ReLU()
            )

        def forward(self, t, x):
            return self.net(x)

    def __init__(self, input_size, output_size):
        super().__init__()

        ode_channels = 1
        ode_dim = input_size[0] * input_size[1] * ode_channels

        self.ode_1 = ODE_Block(self.DiffFunction())
        self.fc1 = nn.Linear(ode_dim, output_size)
        self.conv2d = nn.Conv2d(1, 1, 3, padding="same")

    def forward(self, x):
        x = self.ode_1(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        x = self.fc1(x)
        return x


class InceptionBlock(nn.Module):

    """
    Inplementation of an inception block

    Output tensor is a concatenation of four branches
        1x1 Convolution Layers
        3x3 Convolution Layers
        5x5 Convolution Layers
        MaxPool3d Layers

    param:branch_channels the number of channels in the output of each branch
    """
    
    def __init__(self, branch_channels:int):
        super().__init__()
        in_channels = 4 * branch_channels
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding="same"),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU()
        )

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU()
        )

        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding="same"),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU()
        )

        self.pool = nn.Sequential(

            nn.MaxPool3d(kernel_size=3, stride=(4, 1, 1), padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = torch.cat((
            self.conv_1x1(x),
            self.conv_3x3(x),
            self.conv_5x5(x),
            self.pool(x)),
            dim=1
        )

        return x


class ODE_LeNet(nn.Module):
    """
    ODE block followed by standard LeNet
    The ODE block uses an inception block as its differential function
    """

    class InceptionDiffFunction(nn.Module):
        def __init__(self, inception_branch_channels):
            super().__init__()
            ode_total_channels = 4 * inception_branch_channels
            self.net = InceptionBlock(inception_branch_channels)

        def forward(self, t, x):
            return self.net(x)

    def __init__(self, ODE_channels=12, freezeLeNet=False):
        super().__init__()

        assert ODE_channels % 4 == 0 # our inception implemetation require this

        # choose if our LeNet block is pre-trained
        if freezeLeNet:
            lenet5 = torch.load("lenet")
            lenet5.requires_grad = False
        else:
            lenet5 = LeNet()

        # define our neural net
        self.net = nn.Sequential(
            nn.Conv2d(1, ODE_channels, kernel_size=1, padding="same"),
            nn.BatchNorm2d(ODE_channels),
            ODE_Block(ODE_LeNet.InceptionDiffFunction(ODE_channels//4)),
            nn.BatchNorm2d(ODE_channels),
            nn.Conv2d(ODE_channels, 1, kernel_size=1, padding="same"),
            nn.BatchNorm2d(1),
            lenet5
        )

    def forward(self, x):
        x = self.net(x)
        return x


class LeNet(nn.Module):
    """
    Standard LeNet 5 architecture
    """
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.tanh(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x
