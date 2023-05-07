## Enviroment
1. The environment.yml file to activate env: `conda activate base` 


2. `Dataset` foler: 3 datasets, UCI, HAPT, and HHAR.

##  ========self-supervised learning========

### datasets
 [UCIHAR Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/); 
 [HAPT dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00341/); 
 [HHAR dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00344/)

 [UCIHAR Dataset]https://archive.ics.uci.edu/ml/machine-learning-databases/00240/; 
 
 https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ

 [HAPT dataset]https://archive.ics.uci.edu/ml/machine-learning-databases/00341/; 

 [HHAR dataset]https://archive.ics.uci.edu/ml/machine-learning-databases/00344/; https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO
 
Two ways to pre-process the data:

1. read directly from raw data: 
Download the raw data txt files and save them to 'rawdata' folder. 
Update the data folder path in the main.py file.
some examples in hapt_data/rawdata/

2. convert raw data to .pt files:

2.1. In 'Dataprocessing folder', processiong.py file to generate the train.pt, val.pt and test.pt; 

Move the data.pt to the following folder:

data folder: /HAR-CNN-LSTM/GitFYP_experiment/uci_data 
data folder: /HAR-CNN-LSTM/GitFYP_experiment/hapt_data 
data folder: /HAR-CNN-LSTM/GitFYP_experiment/hhar_data 
 
Rename the 'train.pt' to 'train_100per.py'

2.2. Create 1, 5, 10, 50% train data

create_few_percetages.py to generate the subset of % train data for each dataset; the result will be saved in 'output_data' folder.

Move the data.pt to the corresponding data folder

some examples in uci_data/


### training mode

Supervised: use "supervised" in the training mode.
Self-supervised training: use "ssl"
Fine-tuning: use "ft" in the training mode


### run the code 

###  supervised learning

cd to: 'GitFYP_experiment/supervised/'

Run 'main_pytorch.py'

Result saved in 'result' folder

Data loader: data_preprocess.py

Network: network.py

CNN-LSTM-Attention network: run the main_pytorch.py in 'Attention' folder
CNN-LSTM network: run the main_pytorch.py outside the 'Attention' folder


###  self-supervised learning

cd to: 'GitFYP_experiment/supervised/'

Run 'main.py'

can change paramerter in ArgumentParser() in main.py, e.g.: training mode: ssl or ft or supervised; fine-tuning data percentage, whether oversample, etc.

Result saved in 'result' folder

Data loader: data_loader.py

Network: network.py


**Support Pytorch.**

## Prerequisites
- Python 3.x
- Numpy
- Pytorch 1.0+

There are many public datasets for human activity recognition. You can refer to this survey article [Deep learning for sensor-based activity recognition: a survey](https://arxiv.org/abs/1707.03502) to find more.

In this demo, we will use UCI and HAPT dataset as examples. 


## Code reference
1. supervised learning: https://github.com/jindongwang/Deep-learning-activity-recognition.git
2. self-supervised learning: https://github.com/emadeldeen24/TS-TCC.git; https://github.com/emadeldeen24/eval_ssl_ssc.git





## A CNN-LSTM Approach to Human Activity Recognition in pyTorch with UCI and HAPT dataset

> Deep learning is perhaps the nearest future of human activity recognition. While there are many existing non-deep method, we still want to unleash the full power of deep learning. This repo provides a demo of using deep learning to perform human activity recognition.

In github, there is no repo using **pyTorch nn** with **conv1d and lstm** with UCI and HAPT dataset. 

Since time series data is in 1 dimension, I amended JinDong's network file from conv2d into conv1d. 


## A Self-supervised approach 1D-CNN Approach to Human Activity Recognition in pyTorch

The result is compared the SSL training with few labels and supervised network training with full labels. It proved that SSL can achieved higher f1-score. 





