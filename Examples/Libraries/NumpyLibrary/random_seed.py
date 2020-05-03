import torch
import numpy as np
def xx():
    # np.random.seed(999)
    return np.random.choice([1,2,3,4,5])
def yy():
    yy =torch.rand(10)
    return yy

for i in range(10):
    np.random.seed(999)
    torch.manual_seed(999)
    d = np.random.random((1,2))
    x = np.random.choice([1, 2, 3, 4, 5])
    y = torch.tensor(np.arange(10))
    y = y[torch.randperm(y.shape[0])]
    print('------')
    print(d)
    print(x)
    print(y)
    print(yy())
    # print(xx())


