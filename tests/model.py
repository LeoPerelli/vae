from vae.utils.model import ResidualDecoder, ResidualDecoderBlock, ResidualEncoder, ResidualEncoderBlock
import torch

def test_ResidualDecoder():

    return

def test_ResidualDecoderBlock():

    return

def test_ResidualEncoderBlock():
    
    x_1 = torch.ones(size=(10,3,64,64))
    x_2 = torch.ones(size=(10,64,64,64))

    encoder_1 = ResidualEncoderBlock(in_channels=3, out_channels=64, residual=False) # first block that gets the input
    encoder_2 = ResidualEncoderBlock(in_channels=64, is_first_block=True) # first block when block downsizing images 
    encoder_3 = ResidualEncoderBlock(in_channels=64) # inner block

    y_1 = encoder_1(x_1)
    y_2 = encoder_2(x_2)
    y_3 = encoder_3(x_2)

    v1 = list(y_1.shape) == [10,64,64,64]
    v2 = list(y_2.shape) == [10,128,32,32]
    v3 = list(y_3.shape) == [10,64,64,64]

    return all((v1, v2, v3))

def test_ResidualEncoder():

    x = torch.ones(size=(10,3,256,256))

    encoder = ResidualEncoder(in_channels_start=64, depth=3)

    log_mu, log_sigma = encoder(x)

    v1 = list(log_mu.shape) == [10, 1, 32*32]
    v2 = list(log_sigma.shape) == [10, 1, 32*32]

    return all((v1, v2))
