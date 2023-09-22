import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# Distance metric
def d_m(diff, cov):
    d_metric = np.einsum('...i, ij, ...j->...', diff, cov, diff)
    return d_metric

# Get estimate of volume encompassing the true value with the mean as reference point
def get_percent_volume(samples, truth, cov):
    mean = np.mean(samples, axis=0, keepdims=True)
    volume = np.less(d_m(samples-mean, cov), d_m(truth-mean, cov))
    volume = np.mean(volume, axis=0)
    return volume, mean

# posterior_samples dimensions: [num simulations, num samples, num dimensions]
# true_params dimensions: [num simulations, 1, num dimensions]
def plot_coverage(posterior_samples, true_params, cov, n_points=30):
    percent_volume_set = []
    for samples, true in zip(posterior_samples, true_params):
        volume, mean = get_percent_volume(samples, true, cov)
        percent_volume_set.append(volume)
    percent_volume_set = np.array(percent_volume_set)

    percentages = np.linspace(0.0,1.0,n_points)

    pred_coverage = np.zeros_like(percentages)
    pred_coverage_jn = np.zeros((percent_volume_set.shape[0], n_points))

    for i in range(n_points):
        percent = percentages[i]
        pred_coverage[i] = np.mean(percent_volume_set <= percent)

        for j in range(percent_volume_set.shape[0]):
            samp_percent_volume = np.delete(percent_volume_set, j)
            pred_coverage_jn[j,i] = np.mean(samp_percent_volume <= percent)

    # Estimate the standard deviation from the jacknife
    pred_coverage_std = np.std(pred_coverage_jn, axis=0)

    return percentages, pred_coverage, pred_coverage_std
    
    
def main(dir_name, num_trim_steps):
    
    samples = np.load(os.path.join(dir_name, 'chains.npy'))[:,num_trim_steps:]
    theta = np.load(os.path.join(dir_name, 'thetas.npy'))[:,None,:]
    
    num_dimensions = theta.shape[2]
    num_simulations = theta.shape[0]

    empirical_cov = np.cov(np.transpose(theta.reshape(num_simulations, num_dimensions)))
    empirical_cov = empirical_cov.reshape(num_dimensions,num_dimensions) # reshape as matrix
    
    percentages, pred_coverage, pred_coverage_std = plot_coverage(samples, theta, empirical_cov)
    
    plt.plot(percentages, pred_coverage)
    plt.fill_between(percentages,pred_coverage+pred_coverage_std,pred_coverage-pred_coverage_std,alpha=0.3)
    plt.plot(percentages,percentages,ls='--', color='k')
    
    upper_left_x = min(percentages)
    upper_left_y = max(pred_coverage)
    upper_left_text = 'Underconfident'

    lower_right_x = max(percentages)
    lower_right_y = min(pred_coverage)
    lower_right_text = 'Overconfident'

    plt.text(upper_left_x, upper_left_y, upper_left_text, verticalalignment='top', horizontalalignment='left')
    plt.text(lower_right_x, lower_right_y, lower_right_text, verticalalignment='bottom', horizontalalignment='right')
    plt.xlabel('Percent Probability Volume')
    plt.ylabel('Percent of Examples With True Value in the Volume')
    plt.savefig(os.path.join(dir_name,'coverage.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--dir_name", required=True, help="Directory to use. Defaults to the current directory.")
    parser.add_argument("--num_trim_steps", required=True, type=int, help="Number of initial chain steps to exclude.")
    args = parser.parse_args()
    main(args.dir_name, args.num_trim_steps)
