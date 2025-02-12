# DynaSD_competition
 
Code to run the DynaSD unsupervised seizure annotation and detection algorithm for the AI in epilepsy seizure detection competition. The manuscript is still in preparation but will be made available along with the code upon publication. In brief, the algorithm leverages autoregressive loss as a simple seizure detection feature combined with a threshold learned from annotated intracranial recordigns of spontaneous seizures and spatiotemporal smoothing post processing.

Below is workflow that we used to test and run our containerized algorithm.
```
docker pull ghcr.io/wojemann/dynasd_competition:latest
```

```
docker run -v /path/to/sample/eeg/folder:/data -v /path/to/test/output:/output -e INPUT="sample_eeg.edf" -e OUTPUT="test.tsv" ghcr.io/wojemann/dynasd_competition:latest
```
This outputs the algorithm's seizure annotations as a .tsv file compliant with the AI in epilepsy seizure detection challenge criteria.
