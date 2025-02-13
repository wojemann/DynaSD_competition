from epilepsy2bids.eeg import Eeg
from epilepsy2bids.annotations import Annotations

from timescoring.annotations import Annotation
from timescoring import scoring
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join as ospj
from os import listdir as ls
import sys
sys.path.append('../')
from DSOSD.utils import notch_filter, bandpass_filter, downsample
from DSOSD.model import NDD

cval_param_grid = {
    'smoothing':[(4,5),(9,10),(14,15)],
    'training':[10,30,60],
    'spatial_threshold':[4,6,9],
    'threshold':['cval'],
    'high_pass':[55,100],
    'low_pass':[1,4]
}

root = '/Users/wojemann/local_data/CHBMIT_BIDS/'
subjs = ls(root)
cval_results_list=[]
qbar = tqdm(subjs)
for subj_path in qbar:
    if 'sub' not in subj_path:
        continue
    subj_str = subj_path.split('-')[1]
    qbar.set_description(subj_path)
    runs = ls(ospj(root,subj_path,'ses-01','eeg'))
    pbar = tqdm(runs)
    for run_path in pbar:
        if 'edf' not in run_path:
            continue
        run_str = run_path.split('_')[-2]
        pbar.set_description(run_str)
        eeg = Eeg.loadEdfAutoDetectMontage(edfFile=ospj(root,subj_path,'ses-01','eeg',run_path))
        if eeg.montage is Eeg.Montage.UNIPOLAR:
            eeg.reReferenceToBipolar()
        fs = eeg.fs

        annot_str = run_path[:-8]+'_events.tsv'
        annot = Annotations.loadTsv(ospj(root,subj_path,'ses-01','eeg',annot_str))

        signal_df = pd.DataFrame(eeg.data.T,columns=eeg.channels)
        for low in cval_param_grid['low_pass']:
            for high in cval_param_grid['high_pass']:
                signal_filt = bandpass_filter(notch_filter(signal_df,fs,axis=0),fs,lo=low,hi=high,axis=0)
                signal_ds,fs_ds = downsample(signal_filt,fs,128)
                signal_ds_df = pd.DataFrame(signal_ds,columns=eeg.channels)
                mdl = NDD(fs=fs_ds)
                for train in cval_param_grid['training']:
                    mdl.fit(signal_ds_df.iloc[:train*fs_ds,:])
                    sz_prob = mdl(signal_ds_df)
                    for thresh in cval_param_grid['threshold']:
                        if thresh == 'cval':
                            mdl.get_cval_threshold();
                        else:
                            mdl.get_gaussianx_threshold(sz_prob);
                        for wins in cval_param_grid['smoothing']:
                            _,sz_clf = mdl.get_onset_and_spread(sz_prob,
                                                                rwin_size=wins[1],
                                                                rwin_req=wins[0],
                                                                ret_smooth_mat=True,
                                                                )
                            for space in cval_param_grid['spatial_threshold']:
                                dynasd_mask_wins = sz_clf.sum(axis=1) > space

                                window_start_times = mdl.get_win_times(signal_ds_df.count().median().astype(int))
                                # Convert window start times to sample indices
                                start_sample_indices = (window_start_times * fs_ds).astype(int)
                                end_sample_indices = ((window_start_times + 1) * fs_ds).astype(int)

                                # Implement forward filling in window space
                                dynasd_mask_wins_ff = dynasd_mask_wins.copy()

                                # Ensure a classification is valid if at least `rwin_req_idx` out of `rwin_size_idx` future windows are true
                                for j in range(len(dynasd_mask_wins) - mdl.rwin_size_idx):
                                    if dynasd_mask_wins[j]:  # If window j is classified as true
                                        future_sum = np.sum(dynasd_mask_wins[j:j + mdl.rwin_size_idx])  # Count future true windows
                                        if future_sum >= mdl.rwin_req_idx:  # If requirement met, propagate forward
                                            dynasd_mask_wins_ff[j:j + mdl.rwin_size_idx] = True


                                # Initialize a sample-level seizure array (assume total duration is max end time)
                                sample_length = end_sample_indices[-1]  # Assuming last window extends to end
                                dynasd_mask = np.zeros(sample_length, dtype=bool)

                                # Map window classification to sample space
                                for start, end, is_seizing in zip(start_sample_indices, end_sample_indices, dynasd_mask_wins_ff):
                                    if is_seizing:  # Only mark if the window was classified as seizing
                                        dynasd_mask[start:end] = True
                                
                                mask_true = annot.getMask(fs_ds)
                                plt.figure()
                                plt.plot(mask_true)
                                plt.plot(dynasd_mask)
                                plt.savefig(ospj('./figures',f'{subj_str}_{run_str}_{low}_{high}_{train}_{thresh}_{wins}_{space}.png'))
                                plt.close()
                                hyp = Annotation(dynasd_mask,fs_ds)
                                ref = Annotation(mask_true,fs_ds)
                                param = scoring.EventScoring.Parameters(
                                    toleranceStart=30,
                                    toleranceEnd=60,
                                    minOverlap=0,
                                    maxEventDuration=5 * 60,
                                    minDurationBetweenEvents=90)
                                scores = scoring.EventScoring(ref, hyp, param)
                                cval_results_list.append({
                                    'patient':subj_path,
                                    'run':run_str,
                                    'threshold':thresh,
                                    'low_pass':low,
                                    'high_pass':high,
                                    'training':train,
                                    'smoothing':wins,
                                    'spatial_threshold':space,
                                    'sensitivity': scores.sensitivity,
                                    'precision': scores.precision,
                                    'F1':scores.f1,
                                    'fprate':scores.fpRate
                                })

cval_results_df = pd.DataFrame(cval_results_list)
print(cval_results_df)
cval_results_df.to_pickle('cval_results_df.pkl')