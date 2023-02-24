##  Multi-Connectomes Fusion Network for Major Depressive Disorder Diagnosis


###  Model Architecture


###   Code Description

Our code can be found in the Model-run folder.

- graphcnn/: definitation of the proposed model
- main_gcn_lstm: test the performance of saved models

Run `graphcnn/setup/dti_fmri_pre_process.py` to preprocess the data and run `main_gcn_lstm.py` to test the performance of the proposed method.

### Environment
The code is developed and tested under the following environment

- tensorflow-gpu = 1.15.0
- scikit-learn  >= 0.24.2
- numpy	>= 1.16.6