import jax
import jax.numpy as jnp
from flax import nnx


class Layer(nnx.Module):
    def __init__(
        self,
        din: int,
        dout: int,
        act: nnx.relu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, (din, dout)) / 10)
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.act = act

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
        hidden_activation=nnx.relu,
        output_activation=nnx.identity,
        rngs: nnx.Rngs,
    ):
        key_in, key_hidden1, key_hidden2, key_out = jax.random.split(rngs.params(), 4)
        self.hidden_activation = hidden_activation

        # dynamic layer stack creation
        @nnx.split_rngs(splits=layer_depth - 1)
        @nnx.vmap(in_axes=(0))
        def create_layer(rngs):
            return Layer(
                din=layer_width,
                dout=layer_width,
                act=hidden_activation,
                rngs=rngs,
            )

        self.in_layer = Layer(
            din=num_inputs,
            dout=layer_width,
            act=hidden_activation,
            rngs=nnx.Rngs(params=key_in),
        )
        self.hidden_stack = create_layer(
            rngs=nnx.Rngs(params=key_hidden1),
        )
        # later replaced with AutoEncoder
        self.ae = Layer(
            din=layer_width,
            dout=layer_width,
            act=output_activation,
            rngs=nnx.Rngs(params=key_hidden2),
        )
        self.out_layer = Layer(
            din=layer_width,
            dout=num_outputs,
            act=output_activation,
            rngs=nnx.Rngs(params=key_out),
        )

    def __call__(self, x, get_logits: bool = False):
        x = self.in_layer(x)
        x = hidden_forward(
            carry=x,
            layer=self.hidden_stack,
        )
        x = self.ae(x)
        if get_logits is True:
            return x
        x = self.out_layer(x)
        return x


@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
def hidden_forward(carry, layer: Layer):
    return layer(carry)


class AutoEncoder(nnx.Module):
    def __init__(
        self,
        d_mlp: int = 100,
        d_enc: int = 5000,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_enc = d_enc
        key_enc, key_dec = jax.random.split(rngs.params(), 2)
        initializer = jax.nn.initializers.he_uniform()
        self.W_enc = nnx.Param(
            initializer(key_enc, (d_mlp, d_enc)),
        )
        self.W_dec = nnx.Param(
            initializer(key_dec, (d_enc, d_mlp)),
        )
        self.b_enc = nnx.Param(jnp.zeros((d_enc,)))
        self.b_dec = nnx.Param(jnp.zeros((d_mlp,)))
        self.act = nnx.relu
        self.l1_coeff = 0.005

    def __call__(self, x: jax.Array, get_acts: bool = False) -> jax.Array:
        # implements original version, not updated one
        x = x - self.b_dec
        activations = self.act(x @ self.W_enc + self.b_enc)
        if get_acts is True:
            return activations
        x_rcst = activations @ self.W_dec + self.b_dec
        return x_rcst
