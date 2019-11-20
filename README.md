# OmniScale Networks

An implementation of the paper [Omni-Scale Feature Learning for Person Re-Identification](https://arxiv.org/pdf/1905.00953.pdf). The network architecture has been modified slightly, with the use-case of fine-grained classification in mind. You can find the author's official implementation [here](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py).

## Requirements

The model was tested on a 1080-Ti GPU using
- pytorch == 1.2.0
- torchvision == 0.3.0a0+9168476

## Network Modifications

This implementation has a few deliberate differences from the original:
- The first 7x7 convolution layer is replaced by three 3x3 convolution layers
- An extra average pooling layer and 1x1 convolution are added before the global pooling layer. This is get one more downsampling step in before global pooling, as this model is intended for more general image classification (as opposed to person re-id)
- The 512 to 512 linear layer is left out, and a single linear classification layer is used in its place
