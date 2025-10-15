import numpy as np
from dataclasses import dataclass
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.model_selection import KFold

from .model import Embedder


@dataclass
class Data:
    def __init__(
        self,
        embedder: Embedder,
        batch_size: int = 126,
        n_splits: int = 5,
    ) -> None:
        self.batch_size = batch_size
        self.splits = n_splits
        self.embedder = embedder
        self.labels = {0: "world", 1: "sports", 2: "business", 3: "sci/tech"}

    def preprocess(self, dataset) -> dict:
        texts = []
        labels = []
        for data in dataset.as_numpy_iterator():
            title = data["title"].decode("utf-8")
            description = data["description"].decode("utf-8")
            content = title + " " + description
            tokens = self.embedder.tokenize(content)
            texts.append(tokens)
            labels.append(data["label"])

        return texts, np.array(labels, dtype=np.int32)

    def pipeline(self, x, y, reshuffle: bool = True):
        ds = tf.data.Dataset.from_tensor_slices(
            {
                "content": np.array(x),
                "label": np.array(y),
            }
        )
        if reshuffle is True:
            ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True, seed=42)

        ds = ds.batch(
            batch_size=self.batch_size,
            drop_remainder=True,
        ).prefetch(tf.data.AUTOTUNE)
        return ds

    def load(self) -> None:
        # load
        train_ds, test_ds = tfds.load(
            "ag_news_subset",
            split=[
                "train",
                "test",
            ],
        )

        # transform
        train_tokens, self.y_train = self.preprocess(train_ds)
        test_tokens, y_test = self.preprocess(test_ds)

        # embeds
        self.embedder.build_embedding_matrix(train_tokens + test_tokens)
        self.x_train = self.embedder(train_tokens)
        x_test = self.embedder(test_tokens)

        # generate datasets
        self.test_ds = self.pipeline(
            x=x_test,
            y=y_test,
            reshuffle=False,
        )

    def kfolds(self):
        kf = KFold(n_splits=self.splits, shuffle=True, random_state=42)

        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(self.x_train)):
            x_train_fold, y_train_fold = (
                self.x_train[train_indices],
                self.y_train[train_indices],
            )
            x_val_fold, y_val_fold = (
                self.x_train[val_indices],
                self.y_train[val_indices],
            )
            train_ds_fold = self.pipeline(
                x=x_train_fold,
                y=y_train_fold,
            )
            val_ds_fold = self.pipeline(
                x=x_val_fold,
                y=y_val_fold,
            )
            yield fold_idx + 1, train_ds_fold, val_ds_fold
