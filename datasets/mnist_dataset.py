import torch
from torch.utils.data import Dataloader
from torchvision.transforms import Compose, ToTensor, Normalize

# this will be used to transform the image into
# something that we can feed into our NN
image_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

# datasets basically are a sort of list datastructure we can query for specific images and labels
train_dataset = torchvision.datasets.MNIST(
    "dataset/", train=True, download=True, transform=image_transform
)
test_dataset = torchvision.datasets.MNIST(
    "dataset/", train=False, download=True, transform=image_transform
)
# TODO make it download to somewhere better

# dataloaders handle batching and shuffling so we can just iterate through
# them every epoch to train our dataset
def get_train_dataloader(batch_size=64, shuffle=True, **kwargs):
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def get_test_dataloader(batch_size=1024, shuffle=True, **kwargs):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


# TODO for this file
# 1. add methods to be able to pull out sub-datasets as dictionaries mapping labels
#     to their corrresponding data (maybe like 20 examples per or 50 examples per)
# 2. make sure download goes to the right folders
