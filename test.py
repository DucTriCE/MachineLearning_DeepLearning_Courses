import torch
import numpy as np

B1 = torch.randn((1, 5))
B2 = torch.randn((1, 6))
B3 = torch.randn((1, 7))
B4 = torch.randn((1, 8))

print(B1[0][0])