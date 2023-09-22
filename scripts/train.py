# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """


import os
import flax
import jax
import jax.numpy as jnp
import numpy as np
import logging
import functools
import h5py
import argparse

from scoregen.models import NCSNpp1D 
from scoregen.models import State, init_model, get_config
from scoregen import VESDE, get_step_fn, get_sampler

import optax
import orbax

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

def train(config, workdir):
    """Runs the training pipeline.

    Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
    """

    def get_latest_checkpoint(checkpoint_dir):
        # Get list of all checkpoints
        checkpoints = os.listdir(checkpoint_dir)

        # Remove non-integer named checkpoints if any
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.isdigit()]

        # If there are no valid checkpoints, return None
        if not checkpoints:
          return None

        # Convert to integer and find the maximum (latest)
        checkpoints = [int(ckpt) for ckpt in checkpoints]
        latest_checkpoint = max(checkpoints)

        return latest_checkpoint

    def create_dataset(config):
        batch_size = config.training.batch_size

        with h5py.File(config.data.data_path, 'r') as hf:
            data = hf['data']

            # Check if arr is 2D
            if data.ndim == 2:
                # If it is 2D, add an extra dimension at the last axis
                data = np.expand_dims(data, axis=-1)

            # Calculate the number of batches per epoch
            num_batches = len(data) // batch_size

            for epoch in range(config.training.num_epochs):
                # Create a shuffled index at the beginning of each epoch
                indices = np.arange(len(data))
                np.random.shuffle(indices)

                for i in range(num_batches - 1):
                    # Create a batch of indices and sort it
                    batch_indices = indices[i * batch_size : (i + 1) * batch_size]
                    batch_indices.sort()
                    #yield np.transpose(data[batch_indices], axes=(0,2,1))
                    yield data[batch_indices]

                # Handle the last batch
                #batch_indices = indices[num_batches * batch_size:]
                #batch_indices.sort()
                #yield np.transpose(data[batch_indices], axes=(0,2,1))
                #yield data[batch_indices]






    rng = jax.random.PRNGKey(config.seed)

    # Initialize model.
    rng, step_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = init_model(step_rng, config)

    # Initialize optimizer
    optimizer = optax.chain(optax.clip(config.optim.grad_clip),
                 optax.adam(learning_rate=config.optim.lr)
                 )
    init_opt_state = optimizer.init(initial_params)

    # Initialize State
    state = State(step=0, opt_state=init_opt_state,
                       lr=config.optim.lr,
                       model_state=init_model_state,
                       params=initial_params,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

    # Create checkpoints and samples directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # Create Orbax checkpoint manager
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_dir, orbax_checkpointer, options)

    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    # If there is a checkpoint, restore it
    if latest_checkpoint is not None:
        state = checkpoint_manager.restore(step=latest_checkpoint, items=state)
    else:
        print("No checkpoint found. Training from scratch...")

    #state = checkpoint_manager.restore(step=160, items=state)
    initial_step = int(state.step)

    # Create dataset loader
    train_ds = create_dataset(config)

    # Setup SDE
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5


    # Build train step function
    p_train_step_fn = jax.pmap(get_step_fn(sde, score_model, 
                      train=True, 
                      optimizer=optimizer), axis_name='batch',donate_argnums=1)

    # Replicate the training state to run on multiple devices
    pstate = flax.jax_utils.replicate(state)

    # # Building sampling functions
    # if config.training.snapshot_sampling:
    #     sampling_shape = (config.sampling.batch_size, config.data.data_size, config.data.num_channels)
    #     sampling_fn = get_sampler(sde, score_model, sampling_shape, sampling_eps)

    # #logging.info("Starting training loop at step %d." % (initial_step,))
    logger.info("Starting training loop.")
    logger.info("number of gpus: {}".format(jax.local_device_count()))

    for step, batch in enumerate(train_ds, start=initial_step):
        batch = jnp.array(batch)
        batch = batch.reshape(jax.local_device_count(), 
                              config.training.batch_size//jax.local_device_count(), 
                              config.data.data_size, 
                              config.data.num_channels)

        #rng, next_rng = jax.random.split(rng)
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)

        # Execute one training step
        (_, pstate), ploss = p_train_step_fn((next_rng, pstate), batch)
        loss = flax.jax_utils.unreplicate(ploss).mean()

        if step % config.training.log_freq == 0:
            logger.info("step: %d, training_loss: %.5e" % (step, loss))

        # Checkpoint
        if step != 0 and step % config.training.checkpoint_freq == 0:
            # Save the checkpoint.
            if jax.host_id() == 0:
                saved_state = flax.jax_utils.unreplicate(pstate)
                saved_state = saved_state.replace(rng=rng)
                save_args = flax.training.orbax_utils.save_args_from_target(saved_state)
                checkpoint_manager.save(step // config.training.checkpoint_freq, saved_state, save_kwargs={'save_args': save_args})
                logger.info("checkpoint %d saved." % (step // config.training.checkpoint_freq)) 




parser = argparse.ArgumentParser(description='Train a jax 1D score model')
parser.add_argument('--config',
                    type=str,
                    required=False,
                    default='config.json',
                    help='the path to the config file')
parser.add_argument('--workdir',
                    type=str,
                    required=False,
                    default='run/',
                    help='the working directory')
parser.add_argument('--log',
                    type=str,
                    required=False,
                    default='training.log',
                    help='name of the log file')
args = parser.parse_args()

os.makedirs(args.workdir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(args.workdir + args.log)
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

config = get_config(args.config)
train(config, args.workdir)



