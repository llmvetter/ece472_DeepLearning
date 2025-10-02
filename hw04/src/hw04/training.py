import optax
import structlog
from flax import nnx

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
    loss = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


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


def test(
    model: Classifier,
    data: Data,
):
    log.info("Starting Evaluation")
    correct = 0
    total = 0
    for batch in data.test_ds.as_numpy_iterator():
        predictions = pred_step(model, batch)
        correct_pred = (predictions == batch["label"]).sum()
        correct += correct_pred
        total += len(batch["label"])
    accuracy = correct / total
    log.info(f"Trained Model evaluated on {total} samples")
    log.info(f"Test accuracy: {accuracy:.4f}")
