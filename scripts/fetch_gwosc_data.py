import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from scipy.signal.windows import tukey
import scipy.signal as sl

from glob import glob

import os
import wget
from gwosc.timeline import get_segments
from gwosc.locate import get_urls

import requests
from slicgw.constants import srate, seglen, tukey_alpha, f_ref
#from slicgw.utils import read_data
import json

import argparse

import scipy

            
            
def read_data(path, **kws):
    with h5py.File(path, 'r') as f:
        t0 = f['meta/GPSstart'][()]
        T = f['meta/Duration'][()]
        h = f['strain/Strain'][:]
        dt = T/len(h)
        time = t0 + dt*np.arange(len(h))
        
        dq_mask = f['quality']['simple']['DQmask'][:]
        
        inj_mask = f['quality']['injections']['Injmask'][:]
        
        return pd.Series(h, index=time, **kws), dq_mask, inj_mask
    
    
def main(save_dir, args):

    bulk_start_time = args.t0 - args.duration//2
    bulk_end_time = args.t0 + args.duration//2
    
    if args.operation == 'fetch':
        print('Fetching data.')
        
        url = 'https://www.gw-openscience.org/eventapi/jsonfull/GWTC/'
        with requests.get(url) as r:
            rjson = r.json()

        with open(os.path.join(save_dir, 'gwosc_data.json'), 'w') as file:
            json.dump(rjson, file)
        
        # get time segments of available data within the specified 'bulk' time
        segments = get_segments(f'{args.ifo}_DATA', bulk_start_time, bulk_end_time)

        # get URLs of data files for the above segments
        urls = get_urls(args.ifo, segments[0][0], segments[-1][-1], sample_rate=srate)
        
        # decide whether to download a file that already exists
        force = False
        for url in urls:
            fname = os.path.basename(url)
            file_path = os.path.join(save_dir, 'raw_{}/'.format(args.ifo), fname)
            if not os.path.exists(file_path) or force:
                wget.download(url, file_path)
            
    
    if args.operation == 'preprocess':
        print('Preprocessing data.')
        
        Nsize = args.seglen_upfactor*seglen*srate
        w = tukey(args.seglen_upfactor*seglen*srate, tukey_alpha)
        dt = 1/srate
        f = np.fft.rfftfreq(Nsize, d=dt)
        
        data_set = []
        psd_set = []
        
        fourier_transforms = []
        
        
        for path in sorted(glob(os.path.join(save_dir, 'raw_{}/'.format(args.ifo), '*.hdf5'))):
            full_data, dq_mask, inj_mask = read_data(path)
            
            dq_mask_all = (dq_mask & 0x7F) == 0x7F
            inj_mask_all = (inj_mask & 0x1F) == 0x1F
            num_segments = len(full_data) // Nsize
            
            for i in range(num_segments):

                if np.all(dq_mask_all[args.seglen_upfactor*seglen*i:args.seglen_upfactor*seglen*(i+1)]) and np.all(inj_mask_all[args.seglen_upfactor*seglen*i:args.seglen_upfactor*seglen*(i+1)]):
                    segment = full_data.iloc[i*Nsize:(i+1)*Nsize]
                    fd_segment = np.fft.rfft(segment * w) * dt
                    fourier_transforms.append(fd_segment)
                    
            data_set.append(full_data)
            
            if len(data_set) > args.max_num_hours:
                break
        
        if args.whitening_factor_directory is not None:
            fourier_mean = np.load(os.path.join('../data/', args.whitening_factor_directory, 'fourier_mean_{}.npy'.format(args.ifo)))
            fourier_sigma = np.load(os.path.join('../data/', args.whitening_factor_directory, 'fourier_sigma_{}.npy'.format(args.ifo)))
        else:
            fourier_mean = np.mean(np.array(fourier_transforms), axis=0)
            fourier_sigma = np.std(np.array(fourier_transforms), axis=0)
        
        seg_td_cov_samples = []
        for seg_fd in fourier_transforms:
            seg_fd = (seg_fd - fourier_mean)/fourier_sigma
            seg_td = np.fft.irfft(seg_fd, norm='ortho') * w
            seg_td = seg_td[args.seglen_upfactor*seglen*srate//2 - seglen*srate//2:args.seglen_upfactor*seglen*srate//2 + seglen*srate//2]
            seg_td_cov_samples.append(seg_td)
        seg_td_cov_samples = np.float32(np.array(seg_td_cov_samples))
        seg_td_cov = np.cov(np.transpose(seg_td_cov_samples))
        np.save(os.path.join(save_dir, 'covariance_{}.npy'.format(args.ifo)), seg_td_cov)
        
        
        print('num datasets:', len(data_set))
        
        T = args.seglen_upfactor*seglen
        
        with open(os.path.join(save_dir, 'gwosc_data.json'), 'r') as file:
            rjson = json.load(file)
        true_event_times = sorted([v['GPS'] for v in rjson['events'].values()])
            
        
        output_dataset = []

        for d in data_set:
            # sampling interval
            dt = d.index[1] - d.index[0]
            # segment length
            Nsize = int(round(T/ dt))
            
            # number of segments
            N_segments = int(len(d) / Nsize)
            
            
            for k in range(N_segments):
                segment = d.iloc[k*Nsize:k*Nsize+Nsize]
                no_events = all([(t0 < segment.index[0] or t0 > segment.index[-1]) for t0 in true_event_times])
                no_nans = not segment.isnull().values.any()
                
                if no_events and no_nans:
                        
                    segment_fd = np.fft.rfft(segment * w) * dt
                    segment_fd = (segment_fd - fourier_mean)/fourier_sigma
                    segment_td = np.fft.irfft(segment_fd, norm='ortho') * w
                    
                    output_dataset.append(segment_td[args.seglen_upfactor*seglen*srate//2 - seglen*srate//2:args.seglen_upfactor*seglen*srate//2 + seglen*srate//2])


        print(f"There are {len(output_dataset)} {args.ifo} segments.")
        
        output_dataset = np.float32(np.array(output_dataset))

        np.save(os.path.join(save_dir, 'noise_{}.npy'.format(args.ifo)), output_dataset)
        np.save(os.path.join(save_dir, 'fourier_mean_{}.npy'.format(args.ifo)), fourier_mean)
        np.save(os.path.join(save_dir, 'fourier_sigma_{}.npy'.format(args.ifo)), fourier_sigma)


        plt.figure()
        plt.plot(output_dataset[0])
        plt.xlabel("Time index")
        plt.ylabel("Amplitude")
        plt.savefig(os.path.join(save_dir, 'noise_example_{}.png'.format(args.ifo)))
        plt.close()
        
        plt.figure()
        plt.loglog(f, np.abs(fourier_mean))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Fourier Mean Magnitude")
        plt.savefig(os.path.join(save_dir, 'fourier_mean_{}.png'.format(args.ifo)))
        plt.close()
        
        plt.figure()
        plt.loglog(f, fourier_sigma)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Fourier Sigma")
        plt.savefig(os.path.join(save_dir, 'fourier_sigma_{}.png'.format(args.ifo)))
        plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fetch graviational wave noise data.")
    
    parser.add_argument("--operation", type=str, choices=["fetch", "preprocess"], help="Choice between fetching data or preprocessing already fetched data.")
    parser.add_argument("--output_folder_name", type=str, default="real_white_noise/", help="Output folder to save files in.")
    parser.add_argument("--t0", type=int, default=1126259462, help="t0 is the center of full segment in seconds.")
    parser.add_argument("--duration", type=int, default=1*60*60, help="Duration of segment in seconds.")
    parser.add_argument("--ifo", type=str, choices=["H1", "L1"], default="H1", help="Instrument ('H1' for Hanford, 'L1' for Livingston).")
    parser.add_argument("--seglen_upfactor", type=int, default=2, help="Factor of segment length relative to final cropped segment size.")
    parser.add_argument("--max_num_hours", type=int, default=50, help="Maximum number of hours of data to preprocess.")
    parser.add_argument("--whitening_factor_directory", type=str, default=None, help="The directory containing the average PSD to use for whitening. Defaults to None.")
    args = parser.parse_args()

    args_dict = vars(args)
    
    save_dir = os.path.join('../data/', args.output_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'raw_{}/'.format(args.ifo)), exist_ok=True)
    
    # Save parameters used for inference.
    with open(os.path.join(save_dir, "params_{}.json".format(args.ifo)), 'w') as file:
        json.dump(args_dict, file)
    
    print(f"Inference parameters saved.")
    
    main(save_dir, args)
