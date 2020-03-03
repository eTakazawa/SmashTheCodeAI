import os.path
import sys
import re
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.model_selection import train_test_split

from smash_the_code_net_pytorch import SmashTheCodeNetTorch

def transform_dataset(dataset):
  # [((x1,y1),t1), ((x2,y2), t2), ...] form to [[x1,x2,x3,...], [y1,y2,y3,...], [t1,t2,t3,...]] form
  x0 = []
  x1 = []
  t = []
  for data in dataset:
    x0.append(data[0][0]) # Board
    x1.append(data[0][1]) # Nexts
    t.append(data[1])     # Action
  x0 = np.array(x0)
  x1 = np.array(x1)
  t = np.array(t)
  return x0, x1, t

def create_tensor_dataset(dataset):
  x0, x1, t = transform_dataset(dataset)
  x0 = torch.Tensor(x0)
  x1 = torch.Tensor(x1)
  t  = torch.LongTensor(t)
  return TensorDataset(x0, x1, t)

class SmashTheCodeNetTrainer():
  def __init__(self):
    self.net = SmashTheCodeNetTorch()
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
  
  def run(self, num_epoch, batch_size, dataset, random_state=0):
    # split train/test
    train_dataset, _ = train_test_split(dataset, test_size=0.2, random_state=random_state)
    train_dataset = create_tensor_dataset(train_dataset)
    # Mini-Batch loader
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epoch):  # loop over the dataset multiple times
      running_loss = 0.0
      for i, batch_data in enumerate(data_loader, 0):
        x0set, x1set, labels = batch_data

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(x0set, x1set)

        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
          print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
          running_loss = 0.0

def main(np_dataset_path):
  dataset = np.load(np_dataset_path)
  trainer = SmashTheCodeNetTrainer()

  dataset = dataset[:1000]
  print('len(dataset): {}'.format(len(dataset)))
  trainer.run(100, 10, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--np_dataset_path", help="numpy save file's path", type=str, required=True)
    args = parser.parse_args()

    main(args.np_dataset_path)