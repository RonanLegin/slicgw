import argparse
import numpy as np
import matplotlib.pyplot as plt
import tarp
import os
import glob
import re

        
def load_and_stack(directory, pattern1, pattern2):
    """
    Loads and stacks .npy files from a specified directory that match given base patterns,
    ensuring that the integer numbers in the filenames match between the two patterns.

    Args:
        directory (str): The directory containing the .npy files.
        pattern1 (str): The base name of the first set of files to match, e.g., 'chains'.
        pattern2 (str): The base name of the second set of files to match, e.g., 'theta_true'.

    Returns:
        tuple: Two numpy arrays containing the stacked arrays from the loaded .npy files,
               one for each pattern.
    """
    def get_files_with_indices(pattern):
        file_pattern = os.path.join(directory, f'{pattern}*.npy')
        files = glob.glob(file_pattern)
        regex = rf'{re.escape(pattern)}(\d+)\.npy$'
        return {int(re.search(regex, os.path.basename(file)).group(1)): file
                for file in files if re.search(regex, os.path.basename(file))}

    files1 = get_files_with_indices(pattern1)
    files2 = get_files_with_indices(pattern2)
    
    common_indices = set(files1.keys()).intersection(files2.keys())
    if not common_indices:
        raise FileNotFoundError("No matching indices found between the file sets.")
    print(common_indices)
    def load_files(files, indices):
        all_samples = []
        for index in sorted(indices):
            data = np.load(files[index])
            if len(data.shape) == 4:
                data = data[0]
            all_samples.append(data)
        return np.stack(all_samples, axis=0)
    
    array1 = load_files(files1, common_indices)
    array2 = load_files(files2, common_indices)

    return array1, array2

        
def main(directories, labels):
    
    thinning = 20
    burn_in_steps = -20000

    num_sims_list = []
    for idx, directory in enumerate(directories):
        samples, thetas = load_and_stack(directory, 'chains', 'theta_true')
        num_sims_list.append(thetas.shape[0])
    num_sims = min(num_sims_list) if num_sims_list else None
    num_alpha_bins = num_sims//5
    print(f'Using {num_sims} simulations for all datasets.')


    # Get null hypothesis 
    ecp_null_set = []
    for i in range(10000):
        f = np.random.uniform(low=0., high=1., size=(num_sims,))
        h, alpha = np.histogram(f, density=True, bins=num_alpha_bins, range=(0,1))
        dx = alpha[1] - alpha[0]
        ecp = np.cumsum(h) * dx
        ecp_null_set.append(np.concatenate([[0],ecp]))
    ecp_null_set = np.array(ecp_null_set)
    mean_ecp_null = np.mean(ecp_null_set, axis=0).flatten()
    error_ecp_null = np.std(ecp_null_set, axis=0).flatten()
                             
    
    all_ecp_sets = []
    all_ecp_errors = []
    all_coverage = []
    all_labels = []
    for idx, directory in enumerate(directories):
        samples, thetas = load_and_stack(directory, 'chains', 'theta_true')        
        samples = samples[:num_sims,burn_in_steps:][:,::thinning]
        samples = np.transpose(samples, (1,2,0,3))
        samples = samples.reshape(-1, num_sims,  thetas.shape[1])

        ecp_set, coverage = tarp.get_tarp_coverage(samples, thetas, references='random', metric='euclidean',  num_alpha_bins=num_alpha_bins, bootstrap=True, num_bootstrap=1000, norm=True)
        mean_ecp = np.mean(ecp_set, axis=0).flatten()
        error_ecp = np.std(ecp_set, axis=0).flatten()

        all_ecp_sets.append(mean_ecp)
        all_ecp_errors.append(error_ecp)
        all_coverage.append(coverage)
        all_labels.append(labels[idx] if idx < len(labels) else f'Dataset {idx+1}')

    
    # Plotting all results together
    plt.figure(figsize=(8, 6))
    for mean_ecp, error_ecp, coverage, label in zip(all_ecp_sets, all_ecp_errors, all_coverage, all_labels):
        plt.plot(coverage, mean_ecp, label=label)
        plt.fill_between(coverage, mean_ecp - error_ecp, mean_ecp + error_ecp, alpha=0.6)

    plt.plot(alpha, mean_ecp_null, color='grey', label='Null Hypothesis')
    plt.fill_between(alpha, mean_ecp_null-error_ecp_null, mean_ecp_null+error_ecp_null, alpha=0.4, color='grey')
    plt.fill_between(alpha, mean_ecp_null-error_ecp_null*3, mean_ecp_null+error_ecp_null*3, alpha=0.2, color='grey')

    plt.plot([0., 1.], [0., 1.], 'k--')  # Black diagonal dashed line
    plt.xlabel("Expected Coverage Probability")
    plt.ylabel("Coverage")
    plt.legend()
    #plt.grid(True)
    plt.savefig(os.path.join(directories[0], 'coverage.png'), bbox_inches='tight', dpi=175)
    print(f"Plot saved to {os.path.join(directories[0], 'coverage.png')}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple directories and labels for plotting TARP results.')
    parser.add_argument('--directories', nargs='+', type=str, help='List of directories of the experiments to load samples from.')
    parser.add_argument('--labels', nargs='+', type=str, help='List of labels for each directory, for use in plotting.')
    args = parser.parse_args()

    if len(args.directories) != len(args.labels):
        print("Error: The number of directories and labels must match!")
    else:
        main(args.directories, args.labels)

