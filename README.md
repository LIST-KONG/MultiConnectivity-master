##  Multi-Connectivity Representation Learning Network for Major Depressive Disorder Diagnosis


###  Model Architecture

![model](https://user-images.githubusercontent.com/88756798/221081349-dcbdbaf9-414e-4be3-94f9-3e1ed6f3f33f.png)

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
