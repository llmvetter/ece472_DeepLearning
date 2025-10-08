import optax
import structlog
from flax import nnx
import jax.numpy as jnp

from .model import Classifier
from .config import TrainingSettings
from .data import Data
from .checkpointing import Checkpointer

log = structlog.get_logger()


def loss_fn(model: Classifier, batch) -> float:
    logits = model(batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss


@nnx.jit
def train_step(
    model: Classifier,
    optimizer: nnx.Optimizer,
    batch,
):
    """Performs a single training step."""

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: Classifier, metrics: nnx.MultiMetric, batch):
    logits = model(batch["image"])
    labels = batch["label"]
    loss = loss_fn(model, batch)
    top_5 = top_5_accuracy(logits=logits, labels=labels)
    metrics.update(
        loss=loss,
        logits=logits,
        labels=labels,
        top_5_acc=top_5,
    )


@nnx.jit
def pred_step(model: Classifier, batch):
    logits = model(batch["image"])
    return logits.argmax(axis=1)


def train(
    model: Classifier,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    metrics: nnx.MultiMetric,
    checkpointer: Checkpointer,
) -> None:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())

    for step, batch in enumerate(data.train_ds.as_numpy_iterator()):
        model.train()
        loss = train_step(model, optimizer, batch)

        if step % 20 == 0 and step > 0:
            log.info(
                "Training progress",
                step=step,
                training_loss=float(loss),
            )
            model.eval()
            for eval_batch in data.val_ds.as_numpy_iterator():
                eval_step(model, metrics, eval_batch)
            for metric, value in metrics.compute().items():
                log.info(f"eval_{metric}", metric=value)
            metrics.reset()

        if step % 100 == 0 and step > 0:
            checkpointer.dump(model, step)
            log.info(f"Checkpoint saved successfully at step {step}.")

        if step >= settings.train_steps:
            break

    log.info("Training finished")


def top_5_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    idxs = jnp.argsort(logits, axis=-1)[:, -5:]
    labels_exp = jnp.expand_dims(labels, axis=-1)
    hits = jnp.any(idxs == labels_exp, axis=-1)
    return jnp.mean(hits.astype(jnp.float32))


def test_top5(
    model: Classifier,
    data: Data,
):
    log.info("Starting Evaluation")
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    model.eval()

    for batch in data.test_ds.as_numpy_iterator():
        logits = model(batch["image"])
        labels = batch["label"]
        batch_size = len(labels)
        predictions_top1 = jnp.argmax(logits, axis=-1)
        correct_top1_pred = (predictions_top1 == labels).sum()
        correct_top1 += correct_top1_pred
        top_5_batch_acc = top_5_accuracy(logits=logits, labels=labels)
        correct_top5_hits = int(top_5_batch_acc * batch_size)
        correct_top5 += correct_top5_hits
        total += batch_size

    accuracy_top1 = correct_top1 / total
    accuracy_top5 = correct_top5 / total

    log.info(f"Trained Model evaluated on {total} samples")
    log.info(f"Test Top-1 accuracy: {accuracy_top1:.4f}")
    log.info(f"Test Top-5 accuracy: {accuracy_top5:.4f}")
