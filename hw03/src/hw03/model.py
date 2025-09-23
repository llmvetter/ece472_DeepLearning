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
        # add dynamic creation
        assert len(layer_depths) == len(layer_kernel_sizes)

        self.conv = nnx.Sequential(
            Conv2d(1, 32, kernel_size=(3, 3), rngs=rngs),
            nnx.Dropout(rate=0.025, rngs=rngs),
            partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            Conv2d(32, 64, kernel_size=(3, 3), rngs=rngs),
            partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2)),
        )
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.dropout2(self.linear1(x)))
        x = self.linear2(x)
        return x
