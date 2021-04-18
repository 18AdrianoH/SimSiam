import torch
import numpy

from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.function import solarize


class SolarizedRandomResizedCrop(object):
    """
    ***** What this Does *****
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
    DEFAULT_SOL_MODE = SolarizedRandomResizedCrop.SOL_MODE_RAND

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

        self.random_cropper = RandomResizedCrop(self.size, **kwargs)

    def __call__(self, image, rgb=True):
        if isinstance(image, numpy.ndarray):
            raise NotImplementedError("Please transform into a tensor first.")
        if not isinstance(image, torch.tensor):
            raise ValueError("Your image must be a torch tensor.")

        random_cropped = self.random_cropper(image)
        solarized_random_cropped = self.rand_solarize_tensor(random_cropped, rgb=rgb)
        return solarized_random_cropped

    def set_sol_prob(self, new_prob):
        self.sol_prob = new_prob

    def set_sol_thresh(self, new_thresh):
        self.sol_thresh = new_thresh

    def set_sol_mode(self, new_mode):
        self.sol_mode = new_mode

    def rand_solarize_tensor(self, tensor, rgb=True):
        # NOTE the tensor must be in the range [0, 255] for each of the channels
        # (i.e. RGB or black and white, format is expected); alternatively each channel
        # can be in the form [0, 1], but you must pass rgb=False for this.

        # tensor must be (H x W x C) or (H x W)

        high = 255 if rgb else 1

        if len(tensor.shape) == 3:
            H, W, C = tensor.shape
        elif len(tensor.shape == 2):
            H, W = tensor.shape
            C = None
        else:
            raise ValueError(
                "Tensor has shape {} which is not supported. Pass in (H x W x C) or (H x W)".format(
                    tensor.shape
                )
            )

        if self.sol_prob is None:
            raise RuntimeError(
                "You never initialized sol_prob. Try doing `set_sol_prob` or initializing with `sol_prob=your_float` with your_float in [0, 1]."
            )

        # we prefer to copy to avoid aliasing issues, also you'll want to train multiple times on the same image
        do_solarize = torch.rand((H, W)) < sol_prob

        solarized = torch.zeros((H, W, C)) if C else torch.zeros((H, W))

        # for very big tensors it would be faster to try to vectorize; also the if statement could be
        # pulled out if we were not to vectorize but need a slight performance boost
        for h in range(H):
            for w in range(W):
                if C:
                    for c in range(C):
                        solarized[h, w, c] = (
                            high - tensor[h, w, c]
                            if do_solarize[h, w]
                            else tensor[h, w, c]
                        )
                else:
                    solarized[h, w] = (
                        high - tensor[h, w] if do_solarize[h, w] else tensor[h, w]
                    )

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
    # TODO we need maybe 3-5 different examples with/without channels, with/without rgb
    # and with/without square filters to try and run our transform
    raise NotImplementedError


if __name__ == "__main__":
    test_SolarizedRandomResizedCrop()
    # TODO goals for this file
    # 1. test our implementation of the transforms with 3-5 static examples
    # 2. use it in the mnist thing
    raise NotImplementedError
