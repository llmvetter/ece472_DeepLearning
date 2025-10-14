import optax
import structlog
from flax import nnx

from .config import TrainingSettings
from .data import Data
from .model import MLP

log = structlog.get_logger()


def loss_fn(model: MLP, batch) -> float:
    logits = model(batch["content"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss


@nnx.jit
def train_step(
    model: MLP,
    optimizer: nnx.Optimizer,
    batch,
):
    """Performs a single training step."""

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: MLP, metrics: nnx.MultiMetric, batch):
    logits = model(batch["content"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


def train(
    model: MLP,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    metrics: nnx.MultiMetric,
) -> None:
    """Train the model using SGD."""

    log.info("Starting training", **settings.model_dump())

    for step, batch in enumerate(data.train_ds.as_numpy_iterator()):
        model.train()
        loss = train_step(model, optimizer, batch)

        if step % 50 == 0 and step > 0:
            log.info(
                "Training progress",
                step=step,
                training_loss=float(loss),
            )
        if step % 100 == 0 and step > 0:
            model.eval()
            for eval_batch in data.val_ds.as_numpy_iterator():
                eval_step(model, metrics, eval_batch)
            for metric, value in metrics.compute().items():
                log.info(f"eval_{metric}", metric=value)
            metrics.reset()

    log.info("Training finished")
