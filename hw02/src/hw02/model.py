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
    ) -> None:
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, (din, dout)) / 10)
        self.b = nnx.Param(jnp.zeros((dout,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.w.value + self.b.value


class MLP(nnx.Module):
    def __init__(
        self,
        num_inputs: int = 2,
        layer_width: int = 100,
        layer_depth: int = 10,
        num_outputs: int = 1,
        *,
        hidden_activation=nnx.relu,
        output_activation=nnx.identity,
        rngs: nnx.Rngs,
    ):
        key_in, key_hidden, key_out = jax.random.split(rngs.params(), 3)

        # dynamic layer stack creation
        @nnx.split_rngs(splits=layer_depth)
        @nnx.vmap(in_axes=(0))
        def create_layer(rngs):
            return Layer(din=layer_width, dout=layer_width, rngs=rngs)

        self.in_layer = Layer(
            din=num_inputs,
            dout=layer_width,
            rngs=nnx.Rngs(params=key_in),
        )
        self.hidden_layers = create_layer(
            rngs=nnx.Rngs(params=key_hidden),
        )
        self.out_layer = Layer(
            din=layer_width,
            dout=num_outputs,
            rngs=nnx.Rngs(params=key_out),
        )
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def __call__(self, x):
        x = self.hidden_activation(self.in_layer(x))
        x = self.hidden_activation(
            hidden_forward(
                carry=x,
                layer=self.hidden_layers,
            )
        )
        x = self.output_activation(self.out_layer(x))
        return x


@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
def hidden_forward(carry, layer: Layer):
    return layer(carry)
