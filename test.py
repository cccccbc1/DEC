import numpy as np
import torch

z = torch.tensor([[1.0, 2.0, 3], [3.0, 4.0, 3], [3, 3,3], [1, 3, 3]])
x = torch.tensor([[1.0, 2.0, 3],  [2, 1,3], [1, 3, 1]])
t = torch.tensor([3, 2, 3])
print(((z.unsqueeze(1) - x) * t.unsqueeze(0).unsqueeze(0)))
print(((z.unsqueeze(1) - x) * t))