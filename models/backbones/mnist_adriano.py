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

    MNIST_IMG_SIZE = (1, 28, 28)

    def __init__(self, image_size=None, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.image_size = image_size if image_size else SimpleCNN.MNIST_IMG_SIZE
        if len(self.image_size) == 3:
            self.C, self.H, self.W = self.image_size
        elif len(self.image_size) == 2:
            self.H, self.W = self.image_size
            self.C = None

        self.num_classes = num_classes

        # we'll use stride 1
        self.conv1_kernel = 5
        self.conv2_kernel = 5

        self.conv1_in_c = self.C if self.C else 1
        self.conv1_out_c = 16
        self.conv2_in_c = self.conv1_out_c
        self.conv2_out_c = 32

        # there will be 2 max pools so we lose 4x from both height and width
        # then for the number of edges will be that (the matrix size)
        # times the number of channels
        h_c = (self.H - self.conv1_kernel + 1) // 2
        h_c = (h_c - self.conv2_kernel + 1) // 2

        w_c = (self.W - self.conv1_kernel + 1) // 2
        w_c = (w_c - self.conv2_kernel + 1) // 2

        self.fc1_in = self.conv2_out_c * h_c * w_c
        self.fc1_out = 64
        self.fc2_in = self.fc1_out
        self.fc2_out = self.num_classes

        self.conv1 = nn.Conv2d(
            self.conv1_in_c, self.conv1_out_c, kernel_size=self.conv1_kernel, stride=1
        )

        self.conv2 = nn.Conv2d(
            self.conv2_in_c, self.conv2_out_c, kernel_size=self.conv2_kernel, stride=1
        )

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(self.fc1_in, self.fc1_out)
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
        # print("After conv 1: {}".format(x.shape))
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # print("After max pool and relu: {}".format(x.shape))
        # out_dimension is now (N, conv1_out_c, H / 2, W / 2)

        # in dimension is ibid
        x = self.conv2(x)
        # print("After conv 2: {}".format(x.shape))
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # print("After conv 2 drop, max pool and relu: {}".format(x.shape))
        # out dimension is (N, conv2_out_c, H / 4 , W / 4)

        # shape will now be horizontal
        # shape (1 x (h / 4 x w / 4 x 32)) = (1 x (h x w x 8))
        x = x.view((-1, self.fc1_in))
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.fc2(x)
        return x

    def fc2_weights(self):
        return self.fc2.detach().cpu().clone()

    def train(
        self,
        train_dataloader,
        test_dataloader,
        optimizer=None,
        writer=None,
        device="cpu",
        num_epochs=64,
        train_name="mnist",
        save_dir="mnist_testing/",
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
            print("Epoch {}".format(epoch))

            train_losses.append(0)
            train_accs.append(0)
            test_losses.append(0)
            test_accs.append(0)

            # ugh if only we could have a pointer to an int
            for dataloader, losses, accs, dataloader_name in [
                (train_dataloader, train_losses, train_accs, "train"),
                (test_dataloader, test_losses, test_accs, "test"),
            ]:
                for batch, targets in dataloader:
                    batch, targets = batch.to(device), targets.to(device)

                    optimizer.zero_grad()

                    reps = self(batch)
                    probs = F.log_softmax(reps, dim=1)
                    preds = torch.argmax(probs, 1)

                    loss = F.nll_loss(probs, targets)

                    losses[-1] += loss.item()
                    accs[-1] += torch.mean((preds == targets).type(torch.FloatTensor))

                    if dataloader_name == "train":
                        loss.backward()
                        optimizer.step()

            train_losses[-1] /= len(train_dataloader)
            train_accs[-1] /= len(train_dataloader)

            test_losses[-1] /= len(test_dataloader)
            test_accs[-1] /= len(test_dataloader)

            test_epoch_loss, train_epoch_loss = test_losses[-1], train_losses[-1]
            test_epoch_acc, train_epoch_acc = test_accs[-1], train_accs[-1]

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

            if epoch % 10 == 0 or epoch == 2 or epoch == num_epochs:
                print(
                    "\tAt {} train loss and {} test loss.".format(
                        train_epoch_loss, test_epoch_loss
                    )
                )
                print(
                    "\tAt {} train accuracy and {} test accuracy".format(
                        train_epoch_acc, test_epoch_acc
                    )
                )
                print("")

        torch.save(self, save_dir + "/model.pt")
        torch.save(self.state_dict(), save_dir + "/state_dict.pt")

        return train_losses, test_losses, train_accs, test_accs


def test_mnist():
    import os
    import shutil
    import sys

    sys.path.append("../../")

    from tensorboardX import SummaryWriter
    from datasets.mnist_dataset import get_train_dataloader, get_test_dataloader

    root_dir = "mnist_testing/"
    tensorboard_dir = root_dir + "/tensorboard"

    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
    if os.path.isdir(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
    os.mkdir(tensorboard_dir)

    writer = SummaryWriter(logdir=tensorboard_dir)

    train_dataloader = get_train_dataloader()
    test_dataloader = get_test_dataloader()
    model = SimpleCNN().to("cpu")

    learning_rate = 0.01
    momentum = 0.5
    opt = optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # it's slow... this is just a test!
    model.train(
        train_dataloader,
        test_dataloader,
        optimizer=opt,
        writer=writer,
        save_dir=root_dir,
        num_epochs=11,
    )
    print("Finished training!")


if __name__ == "__main__":
    test_mnist()
    # TODO goals for this file
    # 1. get a test run
    # 2. figure out how to run testing and prediction using main and linear_eval
    # 3. make it possible to extract the features
    # 4. make sure you are using the right folders
