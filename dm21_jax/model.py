import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import Precision


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size: int, out_size: int, *, key: jax.Array):
        self.weight = jax.random.normal(key, (in_size, out_size)) / math.sqrt(in_size)
        self.bias = jnp.zeros(out_size)

    def __call__(self, x: jax.Array) -> jax.Array:
        # NOTE: high precision matmul must be used to match the tensorflow saved
        # model outputs
        # return jnp.dot(x, self.weight) + self.bias
        return jnp.matmul(x, self.weight, precision=Precision.HIGHEST) + self.bias


class LayerNorm(eqx.Module):
    eps: float = eqx.field(static=True)

    gamma: jax.Array
    beta: jax.Array

    def __init__(self, size: int, *, eps: float = 1e-5):
        self.gamma = jnp.ones(size)
        self.beta = jnp.zeros(size)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = x.mean(axis=-1, keepdims=True)
        std = jnp.sqrt(x.var(axis=-1, keepdims=True) + self.eps)
        return (x - mean) / std * self.gamma + self.beta


class ResidualBlock(eqx.Module):
    linear: Linear
    layer_norm: LayerNorm

    def __init__(self, size: int, *, key: jax.Array):
        self.linear = Linear(size, size, key=key)
        self.layer_norm = LayerNorm(size)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.elu(self.layer_norm(self.linear(x) + x))


class MLP(eqx.Module):
    layers: list[ResidualBlock]

    def __init__(self, layer_sizes: list[int], *, key: jax.Array):
        keys = jax.random.split(key, len(layer_sizes))
        self.layers = [ResidualBlock(size, key=k) for size, k in zip(layer_sizes, keys)]

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class NNFunctional(eqx.Module):
    squash_offset: float = eqx.field(static=True)
    sigmoid_scale_factor: float = eqx.field(static=True)

    input_layer: Linear
    mlp: MLP
    output_layer: Linear

    def __init__(
        self,
        input_size: int = 11,
        output_size: int = 3,
        layer_sizes: list[int] = [256, 256, 256, 256, 256, 256],
        squash_offset=1e-4,
        sigmoid_scale_factor=2.0,
        *,
        key: jax.Array,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.input_layer = Linear(input_size, layer_sizes[0], key=k1)
        self.mlp = MLP(layer_sizes, key=k2)
        self.output_layer = Linear(layer_sizes[-1], output_size, key=k3)
        self.squash_offset = squash_offset
        self.sigmoid_scale_factor = sigmoid_scale_factor

    @classmethod
    def from_tf(cls, model_path):
        """
        Loads model weights from original tensorflow checkpoints. Note that the
        variables in the checkpoint must be the in the same order as the pytree
        of the equinox model, which is the case for the DM21 functional.
        """
        import tensorflow as tf

        weights = []
        for v in tf.saved_model.load(model_path).variables:
            weights.append(jnp.array(v.numpy()))

        functional = cls(key=jax.random.key(0))
        leaves, treedef = jax.tree_util.tree_flatten(functional)
        functional = jax.tree_util.tree_unflatten(treedef, weights)
        return functional

    def local_xc(self, data):
        features = jnp.stack(
            [
                data["rho_a"],
                data["rho_b"],
                data["norm_grad"],
                data["norm_grad_a"],
                data["norm_grad_b"],
                data["tau_a"],
                data["tau_b"],
                data["hfx_a"][0],
                data["hfx_a"][1],
                data["hfx_b"][0],
                data["hfx_b"][1],
            ]
        )

        x = jnp.log(jnp.abs(features) + self.squash_offset)
        x = jnp.tanh(self.input_layer(x))
        x = self.mlp(x)
        enhancement_factors = (
            jax.nn.sigmoid(self.output_layer(x) / self.sigmoid_scale_factor)
            * self.sigmoid_scale_factor
        )

        # NB: this does not match the equation for E^LDA in the paper (bottom of
        # pg. 4 in supplemental), which adds the spin up and spin down density
        # prior to exponentiation, however this is correct per inspection of the
        # tensorflow saved model, and matches the original implementation
        # numerically
        e_lda = (
            -2
            * jnp.pi
            * (3 / (4 * jnp.pi)) ** (4 / 3)
            * (data["rho_a"] ** (4 / 3) + data["rho_b"] ** (4 / 3))
        )
        e_hf = data["hfx_a"][0] + data["hfx_b"][0]
        e_hf_omega = data["hfx_a"][1] + data["hfx_b"][1]

        return jnp.sum(enhancement_factors * jnp.stack([e_lda, e_hf, e_hf_omega]))

    def __call__(
        # self, inputs: FunctionalInputs
        self,
        inputs,
    ) -> (
        jax.Array,
        list[jax.Array],
        list[jax.Array],
        list[jax.Array],
        list[jax.Array],
    ):
        rho_a, grad_a_x, grad_a_y, grad_a_z, _, tau_a = jnp.unstack(
            inputs.rho_a, axis=0
        )
        rho_b, grad_b_x, grad_b_y, grad_b_z, _, tau_b = jnp.unstack(
            inputs.rho_b, axis=0
        )

        norm_grad_a = grad_a_x**2 + grad_a_y**2 + grad_a_z**2
        norm_grad_b = grad_b_x**2 + grad_b_y**2 + grad_b_z**2
        grad_sum = (
            (grad_a_x + grad_b_x) ** 2
            + (grad_a_y + grad_b_y) ** 2
            + (grad_a_z + grad_b_z) ** 2
        )

        def xc_and_grads(
            rho_a,
            rho_b,
            norm_grad,
            norm_grad_a,
            norm_grad_b,
            tau_a,
            tau_b,
            hfx_a,
            hfx_b,
            grid_weight,
        ):
            data = {
                "rho_a": rho_a,
                "rho_b": rho_b,
                "norm_grad": norm_grad,
                "norm_grad_a": norm_grad_a,
                "norm_grad_b": norm_grad_b,
                "tau_a": tau_a,
                "tau_b": tau_b,
                "hfx_a": hfx_a,
                "hfx_b": hfx_b,
            }

            xc, grads = jax.value_and_grad(self.local_xc)(data)

            vxc = xc / (rho_a + rho_b + 1e-12)
            vrho = [grads["rho_a"], grads["rho_b"]]
            vsigma = [grads["norm_grad_a"], grads["norm_grad_b"], grads["norm_grad"]]
            vtau = [grads["tau_a"], grads["tau_b"]]

            # NOTE: in the original code they compute the grad wrt weighted vxc,
            # but we can just use the unweighted gradient and multiply by the
            # weight after
            vhfx = [grads["hfx_a"] * grid_weight, grads["hfx_b"] * grid_weight]

            return vxc, vrho, vsigma, vtau, vhfx

        # xc and grads at all grid points
        return jax.vmap(xc_and_grads)(
            rho_a,
            rho_b,
            grad_sum,
            norm_grad_a,
            norm_grad_b,
            tau_a,
            tau_b,
            inputs.hfx_a,
            inputs.hfx_b,
            inputs.grid_weights,
        )
