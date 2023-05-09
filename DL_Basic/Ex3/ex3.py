import torch
torch.manual_seed(100)

"""
    Create the following tensors:
        1. 3D tensor of shape 20x30x40 with all values = 0
        2. 1D tensor containing the even numbers between 10 and 100
"""
# shape_1 = (20,30,40)
# tsr_1 = torch.zeros(shape_1)
# tsr_2 = torch.arange(10, 101, step=2)
# print(tsr_1, "\n", tsr_2)
"""
    x = torch.rand(4, 6)
    Calculate:
        1. Sum of all elements of x
        2. Sum of the columns of x  (result is a 6-element tensor)
        3. Sum of the rows of x   (result is a 4-element tensor)
"""
# x = torch.rand(4, 6)
# sum_1 = torch.sum(x)
# sum_2 = torch.sum(x, axis=0)
# sum_3 = torch.sum(x, axis=1)
# print(x, sum_1, sum_2, sum_3, sep='\n')
"""
    Calculate cosine similarity between 2 1D tensor:
    x = torch.tensor([0.1, 0.3, 2.3, 0.45])
    y = torch.tensor([0.13, 0.23, 2.33, 0.45])
"""
# cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
# x = torch.tensor([0.1, 0.3, 2.3, 0.45])
# y = torch.tensor([0.13, 0.23, 2.33, 0.45])
# print(cos(x,y))
"""
    Calculate cosine similarity between 2 2D tensor:
    x = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                      [-2.2268, 1.9799, 1.5682, 0.5850],
                      [ 1.2289, 0.5043, -0.1625, 1.1403]])
    y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                      [-0.6679, 0.0793, -2.5842, -1.5123],
                      [ 1.1110, -0.1212, 0.0324, 1.1277]])
"""
# cos_0 = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
# cos_1 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
# x = torch.tensor([[0.2714, 1.1430, 1.3997, 0.8788],
#                   [-2.2268, 1.9799, 1.5682, 0.5850],
#                   [1.2289, 0.5043, -0.1625, 1.1403]])
# y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
#                   [-0.6679, 0.0793, -2.5842, -1.5123],
#                   [1.1110, -0.1212, 0.0324, 1.1277]])
# print(cos_0(x,y), cos_1(x,y))
"""
    x = torch.tensor([[ 0,  1],
                      [ 2,  3],
                      [ 4,  5],
                      [ 6,  7],
                      [ 8,  9],
                      [10, 11]])
    Make x become 1D tensor
    Then, make that 1D tensor become 3x4 2D tensor 
"""
# x = torch.tensor([[0, 1],
#                   [2, 3],
#                   [4, 5],
#                   [6, 7],
#                   [8, 9],
#                   [10, 11]])
# x = torch.reshape(x, (1, 12))
# print(x, x.shape)
# x = torch.reshape(x, (3,4))
# print(x, x.shape)
"""
    x = torch.rand(3, 1080, 1920)
    y = torch.rand(3, 720, 1280)
    Do the following tasks:
        1. Make x become 1x3x1080x1920 4D tensor
        2. Make y become 1x3x720x1280 4D tensor
        3. Resize y to make it have the same size as x
        4. Join them to become 2x3x1080x1920 tensor
"""
# x = torch.rand(3, 1080, 1920)
# y = torch.rand(3, 720, 1280)
# x = torch.reshape(x, (1,3,1080,1920))
# y = torch.reshape(y, (1,3,720,1280))
# print(x, y, x.shape, y.shape, sep='\n')
# y = torch.nn.functional.interpolate(y, size=(1080, 1920))
# print(x, y, x.shape, y.shape, sep='\n')
# z = torch.concat((x,y))
# print(z, z.shape)

