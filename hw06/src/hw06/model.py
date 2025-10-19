import jax
from flax import nnx
import jax.numpy as jnp
import numpy as np


class FeedForward(nnx.Module):
    def __init__(self, n_emb: int, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(
                in_features=n_emb,
                out_features=4 * n_emb,
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Linear(
                in_features=4 * n_emb,
                out_features=n_emb,
                rngs=rngs,
            ),
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.net(x)


class Head(nnx.Module):
    def __init__(self, head_size: int, rngs: nnx.Rngs):
        self.K = nnx.Linear(
            in_features=head_size,
            out_features=head_size,
            use_bias=False,
            rngs=rngs,
        )
        self.Q = nnx.Linear(
            in_features=head_size,
            out_features=head_size,
            use_bias=False,
            rngs=rngs,
        )
        self.V = nnx.Linear(
            in_features=head_size,
            out_features=head_size,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, C = x.shape
        key = self.K(x)  # B, T, head_size
        query = self.Q(x)  # B, T, head_size
        value = self.V(x)  # B, T, head_size

        weights = query @ key.transpose((0, -1, -2)) * self.head_size**-0.5  # B, T, T
        tril = jnp.tril(jnp.ones(shape=(T, T), dtype=bool))
        tril - jnp.broadcast_to(tril, (B, T))  # extend mask to batch dimension
        weights = jnp.where(tril, weights, -jnp.inf)
        weights = nnx.softmax(weights, axis=-1)
        out = weights @ value
        return out


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        n_heads: int,
        head_size: int,
        n_embed: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.heads = nnx.List(Head(head_size, rngs) for _ in range(n_heads))
        self.proj = nnx.Linear(
            in_features=n_embed,
            out_features=n_embed,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h(x) for h in self.heads], axis=-1)
        return self.proj(x)


class Block(nnx.Module):
    def __init__(self, n_embed: int, n_heads: int, rngs: nnx.Rngs):
        head_size: int = n_embed // n_heads
        self.att_heads = MultiHeadAttention(
            n_heads=n_heads,
            head_size=head_size,
            n_embed=n_embed,
            rngs=rngs,
        )
        self.ffw = FeedForward(n_emb=n_embed, rngs=rngs)
        self.ln1 = nnx.LayerNorm(num_features=n_embed, rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=n_embed, rngs=rngs)

    def __call__(self, x: jnp.array) -> jnp.array:
        x = x + self.att_heads(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class Decoder(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_blocks: int,
        block_size: int,
        n_heads: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.token_embedding_table = nnx.Embed(
            num_embeddings=vocab_size,
            features=n_embed,
            rngs=rngs,
        )
        self.position_embed_table = nnx.Embed(
            num_embeddings=block_size,
            features=n_embed,
            rngs=rngs,
        )
        blocks = [Block(n_embed, n_heads, rngs) for _ in range(n_blocks)]
        blocks.append(nnx.LayerNorm(num_features=n_embed, rngs=rngs))
        self.blocks = nnx.Sequential(*blocks)
        self.linear = nnx.Linear(
            in_features=n_embed,
            out_features=vocab_size,
            rngs=rngs,
        )

    def __call__(self, idx_sequence: list[int]) -> jnp.ndarray:
        B, T = idx_sequence.shape
        token_embeds = self.token_embedding_table[idx_sequence]
        pos_embeds = self.position_embed_table[jnp.arange(T)]
        emb = token_embeds + pos_embeds
        x = self.blocks(emb)
        out = self.linear(x)
        return out


def count_params(model: nnx.Module) -> int:
    params = nnx.state(model, nnx.Param)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    return int(total_params)
