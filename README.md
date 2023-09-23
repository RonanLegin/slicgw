# slicgw
Learning Non-Gaussian Noise in Gravitational Waves using SLIC

`slicgw` is a Python package developed for learning non-Gaussian noise in gravitational waves using SLIC. This README provides a step-by-step guide to setting up and running the `slicgw` code on the Rusty cluster.

## Installation Steps

### 1. Loading Python Module
On the Rusty cluster, start by loading the Python module:
```bash
module load python
```

### 2. Creating a Virtual Environment
Create a virtual environment named `slicgw`, utilizing system site-packages.
```bash
python -m venv --system-site-packages slicgw
```

### 3. Activating the Virtual Environment
Activate the created virtual environment:
```bash
source slicgw/bin/activate
```

### 4. Upgrading Pip
Upgrade pip to the latest version:
```bash
pip install --upgrade pip
```

### 5. Installing JAX with CUDA Support
Install JAX with CUDA support from the specified URL:
```bash
pip install --upgrade "jax[cuda11_pip]==0.4.14" jaxlib==0.4.14 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 6. Installing Flax from GitHub
Install Flax directly from the GitHub repository (it is important that Flax is installed after JAX to enable GPU support):
```bash
pip install --upgrade git+https://github.com/google/flax.git
```

### 7. Installing RippleGW and Corner
Install `ripplegw` and `corner` packages:
```bash
pip3 install ripplegw corner
```

### 8. Cloning and Installing the Scoregen Package
Install the diffusion model package:
```bash
git clone git@github.com:RonanLegin/scoregen_jax.git
cd scoregen_jax
python setup.py install
```

### 9. Installing slicgw package
Install the `slicgw` package:
```bash
cd slicgw
python setup.py install
```

## Possible Issues and Resolutions
When utilizing `corner`, you might encounter an error. This can likely be resolved by upgrading the `arviz` and `numba` packages:
```bash
pip install arviz --upgrade
pip install numba --upgrade
```
The exact reason for this error is currently unclear, and further investigation may be needed.
