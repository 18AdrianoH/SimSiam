import torch
import random
import numpy
import itertools

from pprint import PrettyPrinter

pp = PrettyPrinter()

from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import solarize


class SolarizedRandomResizedCrop(object):
    """
    This class will return a solarized random crop of an image for a size chosen a priori when you call it.
    You may pass in `size` for the output crop size and `sol_prob` for the probabiltiy that any given pixel
    (independently chosen) is solarized.

    Solarized simply means that the pixels will be inverted as such: x <= 255 - x.
    Note that this will be applied pointwise for each channel, but random picking will occur
    only along width/height. It is uniform and independent.

    You can also pass a `sol_thresh` param to do solarization by threshold
    (i.e. pixels x > thresh will be solarized) but that is not yet implemented. Should be simple with Pytorch's
    functiona's `solarize` command. You must pick between `sol_mode` SOL_MODE_RAND and SOL_MODE_THRESH to
    do solarization. Default is RAND.

    Both crop and output size are expected to be tuples of (height, width). Default (height, width) is (16, 16).
    Default params are used for the random cropping and resizing with pytorch's native transform, so read this for
    more info: https://pytorch.org/vision/stable/transforms.html). You can pass **kwargs additional to those
    I've mentioned above to change this behavior.

    Check out this document: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html,
    to get a sense of how transforms work in pytorch.
    """

    SOL_MODE_RAND = "rand"  # randomly solarize base on a probability
    SOL_MODE_THRESH = "thresh"  # randomly solarize base on

    DEFAULT_OUTPUT_SIZE = (16, 16)
    DEFAULT_SOL_PROB = 0.0

    def __init__(
        self, size=None, sol_prob=None, sol_thresh=None, sol_mode=None, **kwargs
    ):
        self.size = size if size else SolarizedRandomResizedCrop.DEFAULT_OUTPUT_SIZE
        self.sol_mode = (
            sol_mode if sol_mode else SolarizedRandomResizedCrop.SOL_MODE_RAND
        )
        self.check_sol_mode()

        # these can be none, if both are none errors will be thrown on runtime
        self.sol_prob = sol_prob
        self.sol_thresh = sol_thresh

        # NOTE look at scale and ratio: they might be important
        # if len(kwargs) == 0:
        #     kwargs["scale"] = (0.5, 1.0)
        #     kwargs["ratio"] = (1.0, 1.0)

        self.random_cropper = RandomResizedCrop(self.size, **kwargs)
        self._settings = kwargs

    def __call__(self, image, rgb=True):
        # it is expected that your image will have dimensions (C x H x W) or (H x W)
        if isinstance(image, numpy.ndarray):
            raise NotImplementedError("Please transform into a tensor first.")
        if not isinstance(image, torch.Tensor):
            raise ValueError("Your image must be a torch tensor.")

        do_squeeze = False
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
            do_squeeze = True

        random_cropped = self.random_cropper(image)
        solarized = self.rand_solarize_tensor(random_cropped, rgb=rgb)
        output = solarized.squeeze() if do_squeeze else solarized
        # output = output.type(torch.LongTensor) if rgb else output

        return output

    def set_sol_prob(self, new_prob):
        self.sol_prob = new_prob

    def set_sol_thresh(self, new_thresh):
        self.sol_thresh = new_thresh

    def set_sol_mode(self, new_mode):
        self.sol_mode = new_mode

    def set_size(self, new_size):
        self.size = new_size
        self.random_cropper = RandomResizedCrop(self.size, **self._settings)

    def rand_solarize_tensor(self, tensor, rgb=True):
        # NOTE the tensor must be in the range [0, 255] for each of the channels
        # (i.e. RGB or black and white, format is expected); alternatively each channel
        # can be in the form [0, 1], but you must pass rgb=False for this.

        # tensor must be (C x H x W)

        high = 255 if rgb else 1
        C, H, W = tensor.shape

        if self.sol_prob is None:
            raise RuntimeError(
                "You never initialized sol_prob. Try doing `set_sol_prob` or initializing with `sol_prob=your_float` with your_float in [0, 1]."
            )

        do_solarize = (torch.rand((H, W)) < self.sol_prob).type(torch.FloatTensor)
        solarized = (high - tensor) * do_solarize + tensor * (1 - do_solarize)

        return solarized

    def check_sol_mode(self):
        if self.sol_mode == SolarizedRandomResizedCrop.SOL_MODE_THRESH:
            raise NotImplementedError(
                "Solarization mode is threshold, which is not implemented yet."
            )
        elif self.sol_mode != SolarizedRandomResizedCrop.SOL_MODE_RAND:
            raise ValueError(
                "Two valid solarization modes are {} and {}.".format(
                    SolarizedRandomResizedCrop.RAND, SolarizedRandomResizedCrop.THRESH
                )
            )


# a couple minimalist testers to do a sanity check
def test_SolarizedRandomResizedCrop():
    random.seed(0)
    torch.manual_seed(0)
    numpy.random.seed(0)

    zeros_3channels = torch.zeros((3, 28, 28))
    zeros_0channels = torch.zeros((28, 28))

    sol_rand_mode = SolarizedRandomResizedCrop.SOL_MODE_RAND

    # # check that solarization probability 0 does not
    sol_cropper = SolarizedRandomResizedCrop(sol_prob=0.0, sol_mode=sol_rand_mode)
    for t, kwargs in itertools.product(
        [zeros_0channels, zeros_3channels], [{"rgb": True}, {"rgb": False}]
    ):
        sol_cropper.set_size((14, 14))

        s = sol_cropper(t, **kwargs)
        assert torch.max(s) == 0 and torch.min(s) == 0
        assert s.shape[-1] == 14 and s.shape[-2] == 14

        if len(t.shape) == 3:
            assert t.shape[0] == s.shape[0]

    # check that it will invert with probability 1.0 correctly
    sol_cropper.set_sol_prob(1.0)
    for t, kwargs in itertools.product(
        [zeros_0channels, zeros_3channels], [{"rgb": True}, {"rgb": False}]
    ):
        sol_cropper.set_size((14, 14))

        high = 255 if kwargs["rgb"] else 1

        s = sol_cropper(t, **kwargs)
        assert torch.max(s) == high and torch.min(s) == high
        assert s.shape[-1] == 14 and s.shape[-2] == 14

        if len(t.shape) == 3:
            assert t.shape[0] == s.shape[0]

    # these down here you are going to have to eyeball, they seemed
    # to work to me
    sol_cropper.set_sol_prob(0.5)
    sol_cropper.set_size((5, 5))
    t = (torch.rand((3, 8, 8)) * 255).type(torch.LongTensor)
    print("Applying on\n{}".format(t))
    s = sol_cropper(t, rgb=True)
    print("Got\n{}\n".format(s))

    t = torch.rand((8, 8))
    print("Applying on\n{}".format(t))
    s = sol_cropper(t, rgb=False)
    print("Got\n{}\n".format(s))


if __name__ == "__main__":
    test_SolarizedRandomResizedCrop()
