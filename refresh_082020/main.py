import torch



# EXAMPLES:
x = torch.Tensor([5,6])
y = torch.Tensor([2,5])
print(x*y)


z = torch.zeros([3,5])
print(z)
print(z.shape)


v = torch.rand([2,5])
print(v)

# VIEW any tensor as flattened:
print(v.view([1, 10]))

# RE-ASSIGN AS NEW TENSOR:
v = v.view([1, 10])
print(v)



