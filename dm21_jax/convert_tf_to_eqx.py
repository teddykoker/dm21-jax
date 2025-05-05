from argparse import ArgumentParser
from pathlib import Path
import jax
import equinox as eqx
import jax.numpy as jnp
import tensorflow as tf

from dm21_jax.model import NNFunctional


def functional_from_tf(tf_path: str) -> NNFunctional:
    weights = []
    for v in tf.saved_model.load(tf_path).variables:
        weights.append(jnp.array(v.numpy()))

    model = NNFunctional(key=jax.random.key(0))
    leaves, treedef = jax.tree_util.tree_flatten(model)
    model = jax.tree_util.tree_unflatten(treedef, weights)
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tf-checkpoints-path", type=str, required=True)
    args = parser.parse_args()
    tf_checkpoints_path = Path(args.tf_checkpoints_path)

    checkpoints_path = Path(__file__).parent / "checkpoints"
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    for path in tf_checkpoints_path.iterdir():
        if not path.is_dir():
            continue

        functional = functional_from_tf(path)
        eqx.tree_serialise_leaves(checkpoints_path / f"{path.stem}.eqx", functional)
