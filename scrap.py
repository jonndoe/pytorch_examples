import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

for data in trainset:
    print(data)
    break


X, y = data[0][0], data[1][0]

x_batch, y_batch = data[0], data[1]

print(len(x_batch))
print(y_batch)


import matplotlib.pyplot as plt  # pip install matplotlib

plt.imshow(data[0][8].view(28,28))
plt.show()

total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1

print(counter_dict)

# percentage of each digits
for i in counter_dict:
    print(f"{i} {counter_dict[i]/total*100.0}%")

