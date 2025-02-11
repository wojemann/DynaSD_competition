from epilepsy2bids.annotations import Annotations

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('../')
from DSOSD.utils import notch_filter, bandpass_filter, downsample
from DSOSD.model import NDD

def dynasd_detect(eeg):
    fs = eeg.fs
    signal_df = pd.DataFrame(eeg.data.T,columns=eeg.channels)

    signal_filt = bandpass_filter(notch_filter(signal_df,fs,axis=0),fs,lo=1,hi=100,axis=0)
    signal_ds,fs = downsample(signal_filt,fs,128)
    signal_ds_df = pd.DataFrame(signal_ds,columns=eeg.channels)

    mdl = NDD(fs=fs)
    mdl.fit(signal_ds_df.iloc[:60*fs,:])

    sz_prob = mdl(signal_ds_df)
    mdl.get_cval_threshold()
    _,sz_clf = mdl.get_onset_and_spread(sz_prob,
                                    rwin_size=15,
                                    rwin_req=14,
                                    ret_smooth_mat=True,
                                    )
    dynasd_mask_wins = sz_clf.sum(axis=1) > 9

    window_start_times = mdl.get_win_times(signal_ds_df.count().median().astype(int))
    # Convert window start times to sample indices
    start_sample_indices = (window_start_times * fs).astype(int)
    end_sample_indices = ((window_start_times + 1) * fs).astype(int)

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
            
    # print(dynasd_mask_wins.sum())
    # print(dynasd_mask.sum())
    # plt.plot(dynasd_mask)
    # plt.savefig('/output/test_mask.png')

    hyp = Annotations.loadMask(dynasd_mask,fs)

    return hyp