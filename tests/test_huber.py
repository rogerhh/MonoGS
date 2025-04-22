from utils.slam_utils import HuberLoss
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# Make a evenly spaced grid of points from [-1, 1] of torch.float32
x = torch.linspace(-1, 1, 1000, dtype=torch.float32, requires_grad=True)
x_huber = HuberLoss.apply(x, 0.1)

x_huber.backward(torch.ones_like(x_huber))

print(x_huber)
print("x.grad = ", x.grad)
# plot x
plt.plot(x.detach().numpy(), x_huber.detach().numpy(), label="L2")
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label="L2 grad")
plt.savefig("huber_loss2.png")
