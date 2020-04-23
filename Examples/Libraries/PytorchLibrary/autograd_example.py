import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)

out.backward(retain_graph =True)
d = out * out
print(d.grad_fn.next_functions)
d.backward()
print(d.grad)


# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/fashion_mnist_experiment_1')
# with SummaryWriter('runs/test_hist') as writer:
#     for i in range(2):
#         writer.add_histogram('test_hist', x.grad, 2)

# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# writer = SummaryWriter('runs/test_hist')
# for i in range(10):
#     x = np.random.random(1000)
#     writer.add_histogram('distribution centers', x + i, i)
# writer.close()

