import jax
from flax import nnx
from functools import partial


class Conv2d(nnx.Module):
    def __init__(
        self,
        din: int,
        dout: int,
        kernel_size: tuple[int, int],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm = nnx.BatchNorm(dout, rngs=rngs)
        self.act = nnx.relu
        self.layer = nnx.Conv(
            in_features=din,
            out_features=dout,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.act(self.norm(self.layer(x)))


class Classifier(nnx.Module):
    def __init__(
        self,
        input_depth: int,
        layer_depths: list[int],
        layer_kernel_sizes: list[tuple[int, int]],
        num_classes: int,
        rngs: nnx.Rngs,
    ) -> None:
        # check that all layers have kernels
        assert len(layer_depths) == len(layer_kernel_sizes)

        self.pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

        # dynamic conv stack creation
        modules = []
        in_depth = input_depth
        current_height, current_width = 28, 28
        for out_depth, kernel_size in zip(layer_depths, layer_kernel_sizes):
            modules.append(Conv2d(in_depth, out_depth, kernel_size, rngs=rngs))
            modules.append(nnx.Dropout(rate=0.025, rngs=rngs))
            modules.append(self.pool)

            # keep track of sizes
            in_depth = out_depth
            current_height //= 2
            current_width //= 2

        # calc flattened array size (global avg pooling gave bad results)
        flat_size = current_height * current_width * layer_depths[-1]

        # unpack into sequential
        self.conv = nnx.Sequential(*modules)

        # classification head
        self.linear1 = nnx.Linear(flat_size, 256, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.025, rngs=rngs)
        self.linear2 = nnx.Linear(256, num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.dropout(self.linear1(x)))
        x = self.linear2(x)
        return x
