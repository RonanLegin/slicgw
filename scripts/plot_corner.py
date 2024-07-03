import argparse
import numpy as np
import matplotlib.pyplot as plt
import tarp
import os
import glob
import re
import matplotlib.patches as mpatches
import corner

from PIL import Image

        


def main(save_dir, args):
    
    # Load data
    thinning = 1
    burn_in_steps = -20000
    
    # Initialize variables
    chain1 = None
    chain2 = None
    samples = []

    try:
        chain1 = np.load(args.sample1_path)[burn_in_steps:][::thinning]
        print("Chain 1 loaded:", chain1.shape)
    except FileNotFoundError:
        print("Chain 1 file not found.")

    try:
        chain2 = np.load(args.sample2_path)[burn_in_steps:][::thinning]
        print("Chain 2 loaded:", chain2.shape)
    except FileNotFoundError:
        print("Chain 2 file not found.")

    theta_true = np.load(args.theta_true_path)
    ndim = theta_true.shape[-1]

    # Prepare chains if they are loaded
    if chain1 is not None:
        chain1 = chain1.reshape(-1, ndim)
        samples.append(chain1)
    if chain2 is not None:
        chain2 = chain2.reshape(-1, ndim)
        samples.append(chain2)

    # Check if any chain was loaded
    if not samples:
        raise Exception("No chains were loaded. Exiting.")

    # Concatenate the chains for plotting
    samples = np.vstack(samples)
    truths = theta_true.ravel()
    
    theta_label = [r'$\mathcal{M}$', r'$\eta$', r'$\chi_1$', r'$\chi_2$', r'$d_L$', r'$t_c$', r'$\phi_c$']
    labels = [lbl for lbl in theta_label]
    
    contourf_kwargs = {'alpha': 0.5}
    
    fig = plt.figure(figsize=(12, 12))
    if chain1 is not None:
        corner.corner(chain1, fig=fig, color='skyblue', labels=labels, truths=truths, 
                      truth_color='black', hist_kwargs={'density': True}, bins=50, levels=[0.68, 0.99],plot_contours=True, plot_datapoints=False, plot_density=False, fill_contours=True, smooth=1.0)
    if chain2 is not None:
        corner.corner(chain2, fig=fig, color='orange', labels=labels, truths=truths,
                      truth_color='black', hist_kwargs={'density': True}, bins=50, levels=[0.68, 0.99],plot_contours=True, plot_datapoints=False, plot_density=False, fill_contours=True, smooth=1.0)

        
    
        
    # Add legend patches
    patch_list = []
    if chain1 is not None:
        patch_list.append(mpatches.Patch(color='skyblue', label=args.sample1_label))
    if chain2 is not None:
        patch_list.append(mpatches.Patch(color='orange', label=args.sample2_label))
    if patch_list:
        plt.legend(handles=patch_list, loc='upper right')

    # Save and close the figure
    plt.savefig(os.path.join(save_dir, args.output_filename), dpi=175)
    plt.close(fig)

    # Load and manipulate the images
    corner_image = Image.open(os.path.join(save_dir, args.output_filename))
    data_image = Image.open(args.data_png_path)
    new_width = 900
    aspect_ratio = data_image.width / data_image.height
    new_height = int(new_width / aspect_ratio)
    data_image = data_image.resize((new_width, new_height))
    corner_width, corner_height = corner_image.size
    data_width, data_height = data_image.size
    position = (corner_width - data_width - 10, 10)
    corner_image.paste(data_image, position, data_image)
    corner_image.save(os.path.join(save_dir, args.output_filename))

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--save_dir', type=str, help='Directory to save results in.')
    parser.add_argument('--sample1_path', type=str, help='Filepath to sample set 1.')
    parser.add_argument('--sample2_path', type=str, help='Filepath to sample set 2.')
    parser.add_argument('--data_png_path', type=str, help='Filepath to data png.')
    parser.add_argument('--sample1_label', type=str, help='Label for sample1.')
    parser.add_argument('--sample2_label', type=str, help='Label for sample2.')
    parser.add_argument('--theta_label', type=str, nargs='+', help='Label for parameters.')
    parser.add_argument('--output_filename', type=str, help='name for output results.')
    parser.add_argument('--theta_true_path', type=str, help='Filepath to ground truth parameters.')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    main(args.save_dir, args)
