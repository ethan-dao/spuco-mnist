import spuco.datasets as dsets
from spuco.datasets.base_spuco_dataset import SpuriousFeatureDifficulty
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


# 1. Use default parameters to initialize dataset
classes = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
difficulty = SpuriousFeatureDifficulty.MAGNITUDE_LARGE
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])
dataset = dsets.spuco_mnist.SpuCoMNIST(
    root="data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    spurious_correlation_strength=0.5, # High spurious correlation; want to train model to not rely on these spurious features
    transform=transform
)
dataset.initialize()

# Test: Print data set length, classes, and show the data set
# print(len(dataset))
# print(dataset.group_weights)
# plt.imshow(image)
# plt.title(f'Label: {label}')
# plt.show()
image_tensor, label = dataset.data.X[0], dataset.data.labels[0]
print(f"Sample image shape: {image_tensor.shape}")  # Check the shape of the input image

# Testing labels (just in case)
dataset_labels = [label for _, label in dataset]
print(len(dataset_labels))

conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
fc1 = nn.Linear(128*3*3, 512)
fc2 = nn.Linear(512, 128)
fc3 = nn.Linear(128, 10)

# Dimension testing
x = conv1(image_tensor)
print(x.shape)
x = pool(F.relu(x))
print(x.shape)
x = conv2(x)
print(x.shape)
x = pool(F.relu(x))
print(x.shape)
x = conv3(x)
print(x.shape)
x = pool(F.relu(x))
print(x.shape)
x = x.view(-1, 128*3*3) 
print(x.shape)
x = F.relu(fc1(x))
print(x.shape)
x = F.relu(fc2(x))
print(x.shape)
x = fc3(x)
print(x.shape)



