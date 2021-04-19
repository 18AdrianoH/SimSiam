import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

# this will be used to transform the image into
# something that we can feed into our NN
# consider applying: Normalize((0.1307,), (0.3081,)), it was in the code I based this on
image_transform = Compose([ToTensor()])

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


def get_sub_dataset(num_each=20, train=False):
    dataset = train_dataset if train else test_dataset
    sub_dataset = {label: [] for label in range(0, 10)}  # 0 - 9 inclusive

    remaining = set(sub_dataset.keys())
    for data, label in dataset:
        sub = sub_dataset[label]
        if len(sub) >= num_each and label in remaining:
            remaining.remove(label)
        elif len(sub) < num_each:
            sub.append(data)

    # kept as debugging
    # for k, v in sub_dataset.items():
    #     print(k, [vv.shape for vv in v])

    return sub_dataset


# minimalistic debugging here in main
if __name__ == "__main__":
    print("testing get_sub_dataset")
    get_sub_dataset(num_each=4, train=False)
    print("ok!\n")

    print("displaying the dataset characteristics")
    for dataset, name in [(train_dataset, "train"), (test_dataset, "test")]:
        print("dataset", name)
        # 1 x 28 x 28
        # (tensor, label); label is an int
        print("first elem", dataset[0][0].shape, dataset[0][1])

        # labels are 0 - 9
        print("min label", min(map(lambda tp: tp[-1], dataset)))
        print("max label", max(map(lambda tp: tp[-1], dataset)))
        print("dataset size", len(dataset))
        print("ok!\n")

    print("testing dataloaders")
    tdl = get_train_dataloader()
    ttdl = get_test_dataloader()
    print("ok!")
