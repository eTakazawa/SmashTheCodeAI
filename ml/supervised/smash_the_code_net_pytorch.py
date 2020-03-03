import torch
import torch.nn as nn
import torch.optim as optim

from game_const import *

# conv kernel_size=3x3，padding=stride=1
def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                   padding=dilation, groups=groups, bias=True,
                   dilation=dilation)

def conv2x2(in_channels, out_channels, stride=1, groups=1, dilation=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride,
                   padding=dilation, groups=groups, bias=True,
                   dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)

class BasicBlock(nn.Module):
  #  Implementation of Basic Building Block
  def __init__(self, in_channels, out_channels, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_channels, out_channels, stride)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(out_channels, out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample

  def forward(self, x):
    identity_x = x  # hold input for shortcut connection
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity_x = self.downsample(x)

    out += identity_x  # shortcut connection
    return self.relu(out)

class ResidualLayer(nn.Module):
  def __init__(self, num_blocks, in_channels, out_channels, block=BasicBlock):
    super(ResidualLayer, self).__init__()
    downsample = None
    if in_channels != out_channels:
      downsample = nn.Sequential(
        conv1x1(in_channels, out_channels),
        nn.BatchNorm2d(out_channels)
      )
    self.first_block = block(in_channels, out_channels, downsample=downsample)
    self.blocks = nn.ModuleList(block(out_channels, out_channels) for _ in range(num_blocks))

  def forward(self, x):
    out = self.first_block(x)
    for block in self.blocks:
      out = block(out)
    return out

class SmashTheCodeNetTorch(nn.Module):
  def __init__(self):
    super(SmashTheCodeNetTorch, self).__init__()
    
    # Board Input: (B,H,W,C)=(None,12,6,6) (H×W=12×6, C=colors + scull) (C=the number of colors(5) plus the scull(1))
    ## the number of channels (Board)
    bd_in_channels  = NUM_COLORS + 1
    bd_1st_channels = 64
    bd_res_channels = 256
    bd_out_channels = 2
    ## the number of ResBlock
    bd_num_blocks = 14

    ## First layers
    self.bd_conv1 = conv3x3(bd_in_channels, bd_1st_channels)
    self.bd_bn1   = nn.BatchNorm2d(bd_1st_channels)
    self.bd_relu1 = nn.ReLU(inplace=True)
    ## ResNet
    self.bd_reslayer = ResidualLayer(bd_num_blocks, in_channels=bd_1st_channels, out_channels=bd_res_channels)
    self.bd_conv2    = conv1x1(bd_res_channels, bd_out_channels)
    self.bd_bn2      = nn.BatchNorm2d(bd_out_channels)
    self.bd_relu2    = nn.ReLU(inplace=True)
    
    # Nexts input: (B,H,W,C)=(None,8,2,5) (You can see up to 8 puyo)
    ## the number of channels (Next)
    nx_in_channels  = NUM_COLORS
    nx_1st_channels = 64
    nx_res_channels = 256
    nx_out_channels = 2
    ## the number of ResBlock
    nx_num_blocks = 7
    ## First layers
    self.nx_conv1 = conv2x2(nx_in_channels, nx_1st_channels)
    self.nx_bn1   = nn.BatchNorm2d(nx_1st_channels)
    self.nx_relu1 = nn.ReLU(inplace=True)
    ## ResNet
    self.nx_reslayer = ResidualLayer(nx_num_blocks, in_channels=nx_1st_channels, out_channels=nx_res_channels)
    self.nx_conv2    = conv1x1(nx_res_channels, nx_out_channels)
    self.nx_bn2      = nn.BatchNorm2d(nx_out_channels)
    self.nx_relu2    = nn.ReLU(inplace=True)
    
    # Output Layers
    out_mid_features = 256
    out_features = NUM_ACTIONS
    self.out_fc1   = nn.Linear(in_features=198, out_features=out_mid_features)
    self.out_relu1 = nn.ReLU(inplace=True)
    self.out_fc2   = nn.Linear(in_features=out_mid_features, out_features=out_features)
    self.out_relu2 = nn.ReLU(inplace=True)

  # x0: Board, x1: Nexts
  def forward(self, x0, x1):
    x0 = self.bd_conv1(x0)
    x0 = self.bd_bn1(x0)
    x0 = self.bd_relu1(x0)
    x0 = self.bd_reslayer(x0)
    x0 = self.bd_conv2(x0)
    x0 = self.bd_bn2(x0)
    x0 = self.bd_relu2(x0)
    x0 = x0.view(x0.size(0), -1)

    x1 = self.nx_conv1(x1)
    x1 = self.nx_bn1(x1)
    x1 = self.nx_relu1(x1)
    x1 = self.nx_reslayer(x1)
    x1 = self.nx_conv2(x1)
    x1 = self.nx_bn2(x1)
    x1 = self.nx_relu2(x1)
    x1 = x1.view(x1.size(0), -1)

    out = torch.cat((x0, x1), 1)
    # print(out)
    out = self.out_fc1(out)
    out = self.out_relu1(out)
    out = self.out_fc2(out)
    out = self.out_relu2(out)

    return out
