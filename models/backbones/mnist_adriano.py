import tqdm  # TODO use this to create a nicer loading bar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleCNN(nn.Module):
    """
    A very simple convolutional neural network architecture meant to be used for MNIST.

    MNIST is a relatively small dataset so we shouldn't need anything massive (i.e. a big
    ResNet). Also, if we were to use something massive it would likely overfit and we'd need
    to train for longer to learn semantically meaningful features.

    I roughly based it on the first thing I found on Google:
    https://github.com/jiuntian/pytorch-mnist-example/blob/master/pytorch-mnist.py. I then made
    some edits to get it to run nicely for our purposes.
    """

    MNIST_IMG_SIZE = (28, 28)

    def __init__(self, image_size=None, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.image_size = image_size if image_size else SimpleCNN.MNIST_IMG_SIZE
        if len(self.image_size) == 3:
            self.W, self.H, self.C = self.image_size
        elif len(self.image_size) == 2:
            self.W, self.H = self.image_size
            self.C = None

        self.num_classes = num_classes

        self.conv1_in_c = self.C if self.C else 1
        self.conv1_out_c = 16
        self.conv2_in_c = self.conv1_out_c
        self.conv2_out_c = 32

        self.fc1_in = self.conv2_out_c * self.H / 4 * self.W / 4  # TODO
        self.fc1_out = 64
        self.fc2_in = self.fc1_out
        self.fc2_out = self.num_classes

        # convolution layers process the image initially
        # they will use zero padding by default to maintain the dimension
        self.conv1 = nn.Conv2d(
            self.conv1_in_c, self.conv1_out_c, kernel_size=5, stride=1
        )
        self.conv2 = nn.Conv2d(
            self.conv2_in_c, self.conv2_out_c, kernel_size=5, stride=1
        )
        self.conv2_drop = nn.Dropout2d()

        # second fc creates some sort of representation
        self.fc1 = nn.Linear(self.fc1_in, self.fc1_out)

        # second fully connected layer creates the classes
        self.fc2 = nn.Linear(self.fc2_in, self.fc2_out)

    def forward(self, x, reshape=False):
        # NOTE: expects x in shape (N, C, H, W)

        batch_size, C, H, W = x.shape
        if not (C == self.C and H == self.H and W == self.W):
            raise ValueError(
                "Got bad image size. Expecting H x W x C = {} x {} x {}, but got {} x {} x {}.".format(
                    self.H, self.W, self.C, H, W, C
                )
            )

        # in dimension is (N, conv1_in_c, H, W)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # out_dimension is now (N, conv1_out_c, H / 2, W / 2)

        # in dimension is ibid
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # out dimension is (N, conv2_out_c, H / 4 , W / 4)

        # this one will create the "representation"
        x = x.view((-1, self.fc_in))
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x)

        # this last one will create the classes
        x = self.fc2(x)

        return F.log_softmax(x)

    def fc2_weights(self):
        return self.fc2.detach().cpu().clone()

    def reshape_input(x, batch_size=None, C=None, H=None, W=None):
        # NOTE: this will break if the dimensions don't match the original expected dims
        # it is recommended that you pass in C, H, W to confirm
        for expected, gotten, name in filter(
            lambda tp: tp[1], [(self.C, C, "C"), (self.H, H, "H"), (self.W, W, "W")]
        ):
            if expected != gotten:
                raise ValueError(
                    "Got {} for {}, but network is configured for {}".format(
                        gotten, name, expected
                    )
                )

        batch_size = batch_size if batch_size else 1

        return x.view((batch_size, self.C, self.H, self.W))

    def train(
        self,
        train_dataloader,
        test_dataloader,
        optimizer=None,
        writer=None,
        device="cpu",
        num_epochs=64,
        train_name="mnist",
    ):
        # NOTE: look into pytorch lightning: https://www.pytorchlightning.ai/
        # it will make our lives easier in the future if we continue to implement stuff
        self = self.to(device)

        if optimizer is None:
            learning_rate = 0.01
            momentum = 0.5
            optimizer = optim.SGD(
                self.parameters(), lr=learning_rate, momentum=momentum
            )

        train_losses, test_losses = [], []
        train_accs, test_accs = [], []

        for epoch in range(1, num_epochs + 1):
            train_losses.append(0)
            train_accs.append(0)
            test_losses.append(0)
            test_accs.append(0)

            # ugh if only we could have a pointer to an int
            for dataloader, losses, accs in [
                (train_dataloader, train_losses, train_accs),
                (test_dataloader, test_losses, test_accs),
            ]:
                for data, target in dataloader:
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()
                    output = self(data)
                    pred = torch.argmax(output, 1)

                    loss = F.nll_loss(output, target)

                    losses[-1] += loss.item()
                    accs[-1] += torch.mean((pred == target).type(torch.FloatTensor))

                    loss.backward()
                    optimizer.step()

            train_losses[-1] /= len(train_dataloader)
            train_accs[-1] /= len(train_dataloader)

            test_losses[-1] /= len(test_dataloader)
            test_accs[-1] /= len(test_dataloader)

            test_epoch_loss, train_epoch_loss = test_losses[-1], train_losses[-1]
            test_epoch_acc, train_epoch_acc = test_accs[-1], train_losses[-1]

            if writer:
                # the colors might be a tad busted
                writer.add_scalars(
                    "{}/losses/".format(train_name),
                    {"test": test_epoch_loss, "train": train_epoch_loss},
                )
                writer.add_scalars(
                    "{}/accuracies/".format(train_name),
                    {"test": test_epoch_acc, "train": train_epoch_acc},
                )

            if epoch % 10 == 0:
                print("Epoch {}".format(epoch))
                print(
                    "\tAt {} train loss and {} test loss.".format(
                        test_epoch_loss, test_epoch_acc
                    )
                )
                print(
                    "\tAt {} train accuracy and {} test accuracy".format(
                        train_epoch_acc, test_epoch_acc
                    )
                )
                print("")

        # TODO save the parameters or the model somewhere! save the state dict probably
        return train_losses, test_losses, train_accs, test_accs


def test_mnist(self, test_dir=None):
    # NOTE: it's important to test that this model can achieve good results on the MNIST
    # dataset before we start using it as an encoder basically

    import os
    import shutil
    import sys

    sys.path.append("../../")

    from tensorboardX import SummaryWriter
    from datasets.mnist_dataset import train_dataloader, test_dataloader

    root_dir = "../../outputs"
    test_dir = "../../outputs/tensorboard"
    results_dir = "../../outputs/"
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)

    writer = SummaryWriter(logdir=test_dir)

    # TODO implement training and logging while you train
    raise NotImplementedError


if __name__ == "__main__":
    test_mnist(self)
    # TODO goals for this file
    # 1. get a test run
    # 2. figure out how to run testing and prediction using main and linear_eval
    # 3. make it possible to extract the features
    # 4. make sure you are using the right folders
    raise NotImplementedError
