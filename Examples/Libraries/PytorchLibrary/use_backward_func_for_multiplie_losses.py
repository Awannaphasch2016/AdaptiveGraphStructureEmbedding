import torch
import torch.nn as nn
layer = nn.Linear(3, 3)
print(layer.weight.data.fill_(0.1))
layer.weight.grad=None

x1 = torch.ones((3,3), requires_grad=True)
y1=layer(x1)
loss1= torch.sum(y1)
loss1.backward()
print(x1.grad)
print(layer.weight.grad)

x2 = torch.ones((3,3), requires_grad=True)
x2.data.fill_(0.2)
y2=layer(2*x2)
loss2= torch.sum(y2)
loss2.backward()
print(x2.grad)
print(layer.weight.grad)
"""
tensor([[0.3000, 0.3000, 0.3000],
[0.3000, 0.3000, 0.3000],
[0.3000, 0.3000, 0.3000]])
tensor([[3., 3., 3.],
[3., 3., 3.],
[3., 3., 3.]])
tensor([[0.6000, 0.6000, 0.6000],
[0.6000, 0.6000, 0.6000],
[0.6000, 0.6000, 0.6000]])
tensor([[4.2000, 4.2000, 4.2000],
[4.2000, 4.2000, 4.2000],
[4.2000, 4.2000, 4.2000]])
"""

layer = nn.Linear(3, 3)
layer.weight.data.fill_(0.1)

x1 = torch.ones((3,3), requires_grad=True)
x2 = torch.ones((3,3), requires_grad=True)
x2.data.fill_(0.2)

y1=layer(x1)
y2=layer(2*x2)

loss1= torch.sum(y1)
loss1.backward()
print(x1.grad)
print(layer.weight.grad)

loss2= torch.sum(y2)
loss2.backward()
print(x2.grad)
print(layer.weight.grad)
"""
tensor([[0.3000, 0.3000, 0.3000],
[0.3000, 0.3000, 0.3000],
[0.3000, 0.3000, 0.3000]])
tensor([[3., 3., 3.],
[3., 3., 3.],
[3., 3., 3.]])
tensor([[0.6000, 0.6000, 0.6000],
[0.6000, 0.6000, 0.6000],
[0.6000, 0.6000, 0.6000]])
tensor([[4.2000, 4.2000, 4.2000],
[4.2000, 4.2000, 4.2000],
[4.2000, 4.2000, 4.2000]])
"""
layer = nn.Linear(3, 3)
layer.weight.data.fill_(0.1)

x1 = torch.ones((3,3), requires_grad=True)
x2 = torch.ones((3,3), requires_grad=True)
x2.data.fill_(0.2)

y1=layer(x1)
y2=layer(2*x2)

loss1= torch.sum(y1)
loss2= torch.sum(y2)
loss=loss1+loss2
loss.backward()

print(x1.grad)
print(x2.grad)
print(layer.weight.grad)
"""
tensor([[0.3000, 0.3000, 0.3000],
[0.3000, 0.3000, 0.3000],
[0.3000, 0.3000, 0.3000]])
tensor([[0.6000, 0.6000, 0.6000],
[0.6000, 0.6000, 0.6000],
[0.6000, 0.6000, 0.6000]])
tensor([[4.2000, 4.2000, 4.2000],
[4.2000, 4.2000, 4.2000],
[4.2000, 4.2000, 4.2000]])
"""
