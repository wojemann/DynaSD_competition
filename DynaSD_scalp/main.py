import numpy as np
from epilepsy2bids.eeg import Eeg
from .DynaSD_detect import dynasd_detect

def main(edf_file, outFile):
    
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    if eeg.montage is Eeg.Montage.UNIPOLAR:
        eeg.reReferenceToBipolar()

    hyp = dynasd_detect(eeg)

    hyp.saveTsv(outFile)