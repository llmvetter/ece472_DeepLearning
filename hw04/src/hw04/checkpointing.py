import orbax.checkpoint as ocp
from flax import nnx


class Checkpointer:
    def __init__(self):
        self.dir = ocp.test_utils.erase_and_create_empty("/tmp/checkpoints/cifar10/")

    def dump(self, model, step):
        model_state = nnx.state(model)
        with ocp.StandardCheckpointer() as save_checkpointer:
            save_checkpointer.save(
                self.dir,
                state=model_state,
                force=True,
                custom_metadata={"train_step": step},
            )

    def load(self, abstract_model, model_state):
        graphdef, abstract_state = nnx.split(abstract_model)
        with ocp.StandardCheckpointer() as restore_checkpointer:
            restored = restore_checkpointer.restore(self.dir, target=model_state)
        nnx.replace_by_pure_dict(abstract_state, restored)
        model = nnx.merge(graphdef, abstract_state)

        return model
