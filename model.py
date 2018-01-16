import tensorflow as tf
import numpy as np

from tf.contrib.layers import instance_norm
from tf.layers import conv2d, conv2d_transpose
from tf.nn import relu, tanh, leaky_relu

########################################################
# model.py:  A simple conversion from the PyTorch
#   version by yunjey to a TensorFlow implementation
# 
# Original at: https://github.com/yunjey/StarGAN
########################################################

########################################################
# Input Defaults: image_size = 128, g_conv_dim = 64,
#   c_dim=5, d_repeat_num=6
########################################################
def generator(images, image_size, g_conv_dim, c_dim, g_repeat_num):
    """Generator. Encoder-Decoder Architecture."""

    ########################################################
    # Down-sampling
    ########################################################
    gen = relu(instance_norm(conv2d(inputs=images, num_outputs=g_conv_dim, kernel_size=7, stride=1, padding=3)))
    gen = relu(instance_norm(conv2d(inputs=gen, num_outputs=g_conv_dim*2, kernel_size=4, stride=2, padding=1)))
    gen = relu(instance_norm(conv2d(inputs=gen, num_outputs=g_conv_dim*4, kernel_size=4, stride=2, padding=1)))
    
    ########################################################
    # Bottleneck (6 residual blocks)
    ########################################################
    for i in range(g_repeat_num):
        gen = relu(instance_norm(conv2d(inputs=gen, num_outputs=g_conv_dim*4, kernel_size=3, stride=1, padding=1)))

    ########################################################
    # Up-sampling (transpose convolution layers)
    ########################################################
    gen = relu(instance_norm(conv2d_transpose(inputs=gen, num_outputs=g_conv_dim*2, kernal_size=4, stride=2, padding=1)))
    gen = relu(instance_norm(conv2d_transpose(inputs=gen, num_outputs=g_conv_dim, kernal_size=4, stride=2, padding=1)))
    gen = tanh(conv2d(inputs=gen, num_outputs=3, kernal_size=7, stride=1, padding=3))

    return gen

########################################################
# Input Defaults: image_size = 128, g_conv_dim = 64,
#   c_dim=5, d_repeat_num=6
########################################################
def discriminator(images, image_size, d_conv_dim, c_dim, d_repeat_num):

    # Input
    dis = leaky_relu(conv2d(inputs=images, num_outputs=d_conv_dim, kernel_size=4, stride=2, padding=1), alpha=0.01)

    # Hidden
    for i in range(1, d_repeat_num):
        dis = leaky_relu(conv2d(inputs=dis, num_outputs=d_conv_dim*(2**i), kernel_size=4, stride=2, padding=1), alpha=0.01)

    # Output
    dis = conv2d(inputs=dis, num_outputs=1, kernel_size=3, stride=1, padding=1)
    dis = conv2d(inputs=dis, num_outputs=c_dim, kernel_size=image_size*(2**d_repeat_num), stride=1, padding=0)

    return dis
