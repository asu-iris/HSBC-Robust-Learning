import time

import jax
import jax.numpy as jnp
import jax.random as jr
import mujoco
import mujoco.mjx as mjx

def create_hfield(rng, shape) -> jnp.ndarray:
    # dummy implementation
    return jr.uniform(rng, shape, minval=0.0, maxval=1.0)


def create(xml_path: str, batch_size: int):
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mjx_model = mjx.put_model(mj_model)
    mjx_model_batch = jax.tree.map(
        lambda x: x[None].repeat(batch_size, axis=0), mjx_model
    )
    mjx_data_batch = jax.vmap(mjx.make_data)(mjx_model_batch)
    return mjx_model, mjx_model_batch, mjx_data_batch


def reset_(key, mjx_model_batch, mjx_data_batch, reset_mask: jnp.ndarray):
    key = jr.split(key, reset_mask.shape)
    mjx_model_batch = jax.vmap(
        lambda key, reset, mjx_model: (
            jax.lax.cond(
                reset,
                lambda _: mjx_model.replace(
                    hfield_data=create_hfield(key, mjx_model.hfield_data.shape)
                ),
                lambda _: mjx_model,
                operand=None,
            )
        )
    )(key, reset_mask, mjx_model_batch)
    del key

    mjx_data_batch = jax.vmap(
        lambda reset, mjx_model, mjx_data: (
            jax.lax.cond(
                reset,
                lambda _: mjx.make_data(mjx_model),
                lambda _: mjx_data,
                operand=None,
            )
        )
    )(reset_mask, mjx_model_batch, mjx_data_batch)

    return mjx_model_batch, mjx_data_batch


def step_(mjx_model_batch, mjx_data_batch, action):
    mjx_data_batch = mjx_data_batch.replace(ctrl=action)
    mjx_data_batch = jax.vmap(mjx.step)(mjx_model_batch, mjx_data_batch)
    return mjx_data_batch


def main() -> None:
    key = jr.key(42)

    batch_size = 500

    reset = jax.jit(reset_)
    step = jax.jit(step_)

    for idx in range(2):
        print(f"run {idx + 1}")

        t = time.perf_counter()
        mjx_model, mjx_model_batch, mjx_data_batch = create(
            'envs/xmls/env_allegro_cube.xml',
            batch_size,
        )
        print("create", time.perf_counter() - t)

        key, subkey = jr.split(key)
        t = time.perf_counter()
        mjx_model_batch, mjx_data_batch = reset(
            key,
            mjx_model_batch,
            mjx_data_batch,
            jnp.ones((batch_size,), dtype=jnp.bool),
        )
        print("reset all", time.perf_counter() - t)
        del subkey

        ctrl_range = mjx_model.actuator_ctrlrange
        key, subkey = jr.split(key)
        action = jr.uniform(
            subkey,
            (batch_size, ctrl_range.shape[0]),
            minval=ctrl_range[:, 0],
            maxval=ctrl_range[:, 1],
        )
        del subkey

        t = time.perf_counter()
        mjx_data_batch = step(mjx_model_batch, mjx_data_batch, action)
        print("step", time.perf_counter() - t)

        reset_mask = jnp.zeros((batch_size,), dtype=jnp.bool)
        reset_mask = reset_mask.at[: batch_size // 2].set(True)

        key, subkey = jr.split(key)
        t = time.perf_counter()
        mjx_model_batch, mjx_data_batch = reset(
            key, mjx_model_batch, mjx_data_batch, reset_mask
        )
        print("reset some", time.perf_counter() - t)
        del subkey


if __name__ == "__main__":
    main()
