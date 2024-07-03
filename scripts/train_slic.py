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

jax.config.update("jax_enable_x64", False)

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
print('device used: ', xla_bridge.get_backend().platform)

import json

def train(config, args):
    """Runs the training pipeline.

    Args:
    config: Configuration to use.
    args: Arguments from command line.
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

    def random_translate(data, max_shift):
        # Calculate a random shift for each example in the batch
        shifts = np.random.randint(low=0, high=max_shift, size=data.shape[0])
        # Apply periodic boundary condition translation
        translated_data = np.array([np.roll(data[idx], shift) for idx, shift in enumerate(shifts)])
        return translated_data

    def create_dataset(config):
        batch_size = config.training.batch_size
        
        if config.data.ifo == "both":
            filepath_H1 = os.path.join(config.data.data_directory, f"noise_H1.npy")
            filepath_L1 = os.path.join(config.data.data_directory, f"noise_L1.npy")
            data_H1 = np.float32(np.load(filepath_H1))
            data_L1 = np.float32(np.load(filepath_L1))
            
            # Ensure equal contribution from both detectors
            min_length = min(len(data_H1), len(data_L1))
            data_H1 = data_H1[:min_length]
            data_L1 = data_L1[:min_length]
            
            data = np.concatenate((data_H1, data_L1), axis=0)
        else:
            filepath = os.path.join(config.data.data_directory, f"noise_{config.data.ifo}.npy")
            data = np.float32(np.load(filepath))
        
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
                
                # Fetch the data for the current batch
                batch_data = data[batch_indices]
    
                if config.training.random_translation:
                    # Apply random translation with periodic boundaries
                    # Assuming max_shift could be up to feature_size to cover full range of translations
                    batch_data = random_translate(batch_data, max_shift=batch_data.shape[1])
        
                yield batch_data


    rng = jax.random.PRNGKey(config.seed)

    # Initialize model.
    rng, step_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = init_model(step_rng, config)
    
    
    # Initialize optimizer
    if config.optim.optimizer == 'adamax':
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.optim.grad_clip),
            optax.adamaxw(learning_rate=config.optim.lr, weight_decay=config.optim.weight_decay)
        )
    elif config.optim.optimizer == 'adam':
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.optim.grad_clip),
            optax.adamw(learning_rate=config.optim.lr, weight_decay=config.optim.weight_decay)
        )
    else:
        raise ValueError(f"Optimizer '{config.optim.optimizer}' not supported")
        
        
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
    checkpoint_dir = os.path.join(args.workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    sample_dir = os.path.join(args.workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    inference_dir = os.path.join(args.workdir, "inference")
    os.makedirs(inference_dir, exist_ok=True)

    # Create Orbax checkpoint manager
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=None, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_dir, orbax_checkpointer, options)

    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    # If there is a checkpoint, restore it
    if latest_checkpoint is not None:
        state = checkpoint_manager.restore(step=latest_checkpoint, items=state)
    else:
        print("No checkpoint found. Training from scratch...")
    
    initial_step = int(state.step)

    # Create dataset loader
    train_ds = create_dataset(config)

    # Setup SDE
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    
    
    # # Build train step function
    p_train_step_fn = jax.pmap(get_step_fn(sde, score_model, 
                  train=True, 
                  optimizer=optimizer), axis_name='batch',donate_argnums=1)

        
    # Replicate the training state to run on multiple devices
    pstate = flax.jax_utils.replicate(state)

    # #logging.info("Starting training loop at step %d." % (initial_step,))
    logger.info("Starting training loop.")
    logger.info("number of gpus: {}".format(jax.local_device_count()))
    
    for step, batch in enumerate(train_ds, start=initial_step):
        batch = jnp.array(batch, dtype=jnp.float32)
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
        if step % config.training.checkpoint_freq == 0:
            # Save the checkpoint.
            if jax.host_id() == 0:
                saved_state = flax.jax_utils.unreplicate(pstate)
                saved_state = saved_state.replace(rng=rng)
                save_args = flax.training.orbax_utils.save_args_from_target(saved_state)
                checkpoint_manager.save(step, saved_state, save_kwargs={'save_args': save_args})
                logger.info("checkpoint %d saved." % (step)) 




parser = argparse.ArgumentParser(description='Train a jax 1D score model')
parser.add_argument('--config_path',
                    type=str,
                    required=False,
                    default='../data/config.json',
                    help='the path to the config file')
parser.add_argument('--workdir',
                    type=str,
                    required=False,
                    default='/home/ronan/scratch/slicgw/scripts/run_slic/',
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

config = get_config(args.config_path)

with open(args.config_path, 'r') as file:
    data = json.load(file)
with open(os.path.join(args.workdir, 'config.json'), 'w') as file:
    json.dump(data, file)
            
train(config, args)



