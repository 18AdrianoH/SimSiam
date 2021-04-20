from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
from .adriano_aug import MNIST_Transform
def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None):

    if train==True:
        if name == 'simsiam_mnist':
            augmentation = MNIST_Transform(sol_prob=0.5, sol_mode="rand", single=False)
        elif name == 'simsiam':
            augmentation = SimSiamTransform(image_size)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        else:
            raise NotImplementedError("Got name {}".format(name))
    elif train==False:
        if train_classifier is None:
            raise Exception
        # I'm starting to think they really didn't want anyone building on their code
        if name == 'simsiam_mnist':
            augmentation = MNIST_Transform(sol_prob=0.5, sol_mode="rand", single=True)
        else:
            augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








