import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class GroupNorm(nnx.Module):
    def __init__(
        self,
        num_channels: int,
        num_groups: int = 8,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.layer = nnx.GroupNorm(
            num_groups=num_groups,
            num_features=num_channels,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.layer(x)


class Conv2d(nnx.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        kernel_size: tuple[int, int],
        strides: tuple = (1, 1),
        *,
        rngs: nnx.Rngs,
    ) -> None:
        # maybe add kernel init
        self.layer = nnx.Conv(
            in_features=d_in,
            out_features=d_out,
            kernel_size=kernel_size,
            padding="SAME",
            strides=strides,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.layer(x)


class ResidualBlock(nnx.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        kernel_size: int = 2,
        downsample: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.subsample = downsample
        self.conv1 = Conv2d(
            d_in=d_in,
            d_out=d_out,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1) if not self.subsample else (2, 2),
            rngs=rngs,
        )
        self.conv2 = Conv2d(
            d_in=d_out,
            d_out=d_out,
            kernel_size=(kernel_size, kernel_size),
            rngs=rngs,
        )
        self.norm1 = GroupNorm(
            num_channels=d_in,
            num_groups=4,
            rngs=rngs,
        )
        self.norm2 = GroupNorm(
            num_channels=d_out,
            num_groups=4,
            rngs=rngs,
        )
        self.act = nnx.relu
        if self.subsample or d_in != d_out:
            self.downsample = Conv2d(
                d_in=d_in,
                d_out=d_out,
                kernel_size=(1, 1),
                strides=(2, 2) if self.subsample else (1, 1),
                rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        identity = self.downsample(x) if self.subsample else x
        z = self.norm1(x)
        z = self.act(z)
        z = self.conv1(z)
        z = self.norm2(z)
        z = self.act(z)
        z = self.conv2(z)
        return z + identity


class Classifier(nnx.Module):
    def __init__(
        self,
        input_depth: int = 3,
        num_blocks: tuple = (3, 3, 3),
        layer_depths: tuple[int] = (16, 32, 64),
        layer_kernel_sizes: tuple[int] = (3, 3, 3),
        num_classes: int = 10,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """
        Creates a ResNet based classifier.
        Kernels are assumed symmetrical.
        Blocks consist of identical layers.

        """
        # check that all layers have kernels
        assert len(layer_depths) == len(layer_kernel_sizes) == len(layer_kernel_sizes)

        # dynamic conv stack creation
        modules = []

        # input layer
        self.in_layer = Conv2d(
            d_in=input_depth,
            d_out=layer_depths[0],
            kernel_size=(3, 3),
            rngs=rngs,
        )

        in_depth = layer_depths[0]
        # Create ResNet blocks
        for block_idx, (n_layers, out_depth, kernel_size) in enumerate(
            zip(num_blocks, layer_depths, layer_kernel_sizes)
        ):
            for i in range(n_layers):
                downsample = i == 0 and block_idx > 0
                block = ResidualBlock(
                    d_in=in_depth,
                    d_out=out_depth,
                    kernel_size=kernel_size,
                    downsample=downsample,
                    rngs=rngs,
                )
                modules.append(block)
                in_depth = out_depth

        # unpack into sequential
        self.conv_stack = nnx.Sequential(*modules)

        # classification head
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.linear = nnx.Linear(in_depth, num_classes, rngs=rngs)

    def __call__(self, x):
        z = self.in_layer(x)
        z = self.conv_stack(z)
        z = jnp.mean(z, axis=(1, 2))
        z = self.dropout(z)
        z = self.linear(z)
        return z


def count_params(model: Classifier) -> int:
    params = nnx.state(model, nnx.Param)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    return int(total_params)
