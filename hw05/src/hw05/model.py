import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import gensim.downloader as api
from gensim.utils import simple_preprocess


class Embedder:
    def __init__(self, max_words: int = 50):
        self.glove_model = api.load("glove-twitter-25")
        self.vector_size = self.glove_model.vector_size
        self.word_to_index = {"<PAD>": 0}
        self.max_words = max_words

    def tokenize(self, sequence: str) -> list[str]:
        return simple_preprocess(sequence)

    def build_embedding_matrix(self, tokens: list[str]):
        # build mapping word -> idx
        for doc_tokens in tokens:
            for word in doc_tokens:
                if word not in self.word_to_index and word in self.glove_model:
                    self.word_to_index[word] = len(self.word_to_index)
        self.vocab_size = len(self.word_to_index)
        # build mapping idx -> vector
        embedding_matrix = np.zeros((self.vocab_size, 25))
        for word, i in self.word_to_index.items():
            if word in self.glove_model:  # skip pad token
                embedding_matrix[i] = self.glove_model[word]
        self.embedding_matrix = jnp.array(embedding_matrix, dtype=jnp.float32)

    def _index(self, tokens: list[list[str]]) -> list[int]:
        indexed_data = []

        for doc_tokens in tokens:
            indices = [self.word_to_index.get(word, 0) for word in doc_tokens]
            padded_indices = indices[: self.max_words] + [0] * (
                self.max_words - len(indices)
            )
            indexed_data.append(padded_indices)
        return indexed_data

    def __call__(self, sequence: list[list[str]]) -> jnp.ndarray:
        indexed_array = jnp.array(self._index(sequence), dtype=jnp.int32)
        document_vectors = self.embedding_matrix[indexed_array]
        return jnp.mean(document_vectors, axis=1)


class Layer(nnx.Module):
    def __init__(
        self,
        din: int,
        dout: int,
        act=nnx.relu,
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
        num_outputs: int = 4,
        *,
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
            act=output_activation,
            rngs=nnx.Rngs(params=key_in),
        )

    def __call__(self, x):
        x = self.in_layer(x)
        x = hidden_forward(
            carry=x,
            layer=self.hidden_layers,
        )
        x = self.out_layer(x)
        return x


@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
def hidden_forward(carry, layer: Layer):
    return layer(carry)
