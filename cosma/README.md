# cosma

This package contains the implementation of the **spectral** **Co**nvolutional **S**emiregular **M**esh **A**utoencoder.

## Overview
| Module | Contents |
| --- | --- |
| [globals](globals.py) | some frequently used constant variables mainly for creating adjacency matrices, pooling masks and reindexing patches |
| [encoding](encoding.py) | two different pooling operators (```AvgPooler``` and ```IndexPooler```) as well as an encoding block (```Enblock```) combining a pooling operation with a Chebychev convolutional layer |
| [decoding](decoding.py) | unpooling operators (```AvgUnpooler``` and ```PadLevel2IdUnpooler``` (the last unpooler is specifically designed for the unpooling of padded patches)) as well as an decoding block (```Deblock```) combining an unpooling operation with a Chebychev convolutional layer |
| [autoencoder](autoencoder.py) | the actual autoencoder (```CoSMA```) combining multiple encoding blocks to get a low dimensional representation and multiple decoding blocks to reconstruct the original representation from the low dimensional representation|