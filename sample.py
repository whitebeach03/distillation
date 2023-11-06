import torch
import torch.nn as nn

loss = nn.MSELoss()
result = 0
a = torch.randn(3, 3, 3)
b = torch.randn(3, 3, 3)

# 1
for i in range(3):
    res = loss(a[i], b[i])
    result += res
print(result)


closs = loss(a, b)
print(closs)