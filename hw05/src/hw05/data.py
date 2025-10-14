import numpy as np
from dataclasses import dataclass
import tensorflow_datasets as tfds
import tensorflow as tf
from gensim.utils import simple_preprocess

from .model import Embedder


@dataclass
class Data:
    def __init__(
        self,
        batch_size: int = 126,
        train_steps: int = 5000,
        val_split: float = 0.1,
    ) -> None:
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.split = int((1 - val_split) * 100)
        self.embedder = Embedder()

    def preprocess(self, dataset) -> dict:
        texts = []
        labels = []
        for data in dataset.as_numpy_iterator():
            title = data["title"].decode("utf-8")
            description = data["description"].decode("utf-8")
            content = title + " " + description
            tokens = simple_preprocess(content)
            texts.append(tokens)
            labels.append(data["label"])

        return texts, np.array(labels, dtype=np.int32)

    def load(self) -> None:
        # load
        train_ds, val_ds, test_ds = tfds.load(
            "ag_news_subset",
            split=[
                f"train[:{self.split}%]",
                f"train[{self.split}%:]",
                "test",
            ],
        )

        # transform
        train_tokens, y_train = self.preprocess(train_ds)
        eval_tokens, y_eval = self.preprocess(val_ds)
        test_tokens, y_test = self.preprocess(test_ds)

        # embeds
        self.embedder.build_embedding_matrix(eval_tokens + train_tokens + test_tokens)
        x_train = self.embedder(train_tokens)
        x_eval = self.embedder(eval_tokens)
        x_test = self.embedder(test_tokens)

        # generate datasets
        train_ds = tf.data.Dataset.from_tensor_slices(
            {
                "content": np.array(x_train),
                "label": np.array(y_train),
            }
        )
        val_ds = tf.data.Dataset.from_tensor_slices(
            {
                "content": np.array(x_eval),
                "label": np.array(y_eval),
            }
        )
        test_ds = tf.data.Dataset.from_tensor_slices(
            {
                "content": np.array(x_test),
                "label": np.array(y_test),
            }
        )

        # prepare (from jax docu)
        self.train_ds = (
            train_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True, seed=42)
            .batch(
                batch_size=self.batch_size,
                drop_remainder=True,
            )
            .prefetch(tf.data.AUTOTUNE)
        )
        self.val_ds = val_ds.batch(
            batch_size=self.batch_size,
            drop_remainder=True,
        ).prefetch(tf.data.AUTOTUNE)
        self.test_ds = test_ds.batch(
            self.batch_size,
            drop_remainder=True,
        ).prefetch(tf.data.AUTOTUNE)
