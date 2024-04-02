import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LRModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear=nn.Linear(1, 1)

  def forward(self, x):
    return self.linear(x)

class MLRModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear=nn.Linear(3, 1)

  def forward(self, x):
    return self.linear(x)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model=LRModel()

optimizer=optim.SGD(model.parameters(), lr=0.01)

epoch=2000

for i in range(epoch+1):
  P=model(x_train)

  C=F.mse_loss(P, y_train)

  optimizer.zero_grad()
  C.backward()
  optimizer.step()

  if i % 100 == 0:
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          i, epoch, C.item()
      ))