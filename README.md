# CASENet PyTorch Implementation
This is an implementation for paper [CASENet: Deep Category-Aware Semantic Edge Detection](https://arxiv.org/abs/1705.09759).
## Data Processing
We used the data preprocessing codes provided by the author.[sbd-preprocess](https://github.com/Chrisding/sbd-preprocess). Based on this, we generated hdf5 file for data loading. The codes for generating data files are in utils/ folder. We also provided codes to generate png for each class to store the binary value of edge information. (Maybe faster than hdf5 and more efficient in storage. If using this format, need to change the data loader in dataloader/ folder and change the codes for visualization.) 
## Model
Our implemented model is the ResNet101 version of CASENet. We evaluated the official model provided by the author, got a reasonable result but didn't reproduce the same number as in the paper. (The official model is converted from .caffemodel to PyTorch. About the conversion, we first converted the .caffemodel to numpy array, then based on the layername to load corresponding layer into PyTorch model. Codes for the transformation are in modules/CASENet.py)
## Training
Running main.py for training a model.
## Visualization
We modified the visualization codes from [CASENet](http://www.merl.com/research/license#CASENet). PLease see vis_features.py.
## Evaluation Metric
We used the official benchmark codes for evaluation provided by [Semantic Contours from Inverse Detectors](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf). To generate the results image for evaluation, please see get_results_for_benchmark.py.  
