import jax
import jax.numpy as jnp
from flax import nnx


class Layer(nnx.Module):
    def __init__(
        self,
        din: int,
        dout: int,
        *,
        rngs: nnx.Rngs,
        activation: nnx.identity,
    ) -> None:
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.act = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.act(x @ self.w.value + self.b.value)


class MLP(nnx.Module):
    def __init__(
        self,
        num_inputs: int = 2,
        layer_width: int = 100,
        layer_depth: int = 10,
        num_outputs: int = 1,
        *,
        hidden_activation=nnx.identity,
        output_activation=nnx.identity,
        rngs: nnx.Rngs,
    ):
        # dimensions = [num_inputs] + [layer_width] * layer_depth[0]
        # hidden_dims = list(zip(dimensions[:-1], dimensions[1:]))

        # dynamic layer stack creation
        @nnx.split_rngs(splits=layer_depth)
        @nnx.vmap(in_axes=(None, None, 0, None))
        def create_layer(din, dout, rngs, activation):
            return Layer(din, dout, rngs=rngs, activation=activation)

        self.in_layer = Layer(
            din=num_inputs,
            dout=layer_width,
            rngs=rngs,
            activation=hidden_activation,
        )
        self.hidden_layers = create_layer(
            din=layer_width,
            dout=layer_width,
            rngs=rngs,
            activation=hidden_activation,
        )
        self.out_layer = Layer(
            layer_width,
            num_outputs,
            rngs=rngs,
            activation=output_activation,
        )

    def __call__(self, x):
        x = self.in_layer(x)
        x = hidden_forward(
            carry=x,
            layer=self.hidden_layers,
        )
        return self.out_layer(x)


@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
def hidden_forward(carry, layer: Layer):
    return layer(carry)
