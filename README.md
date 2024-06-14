# Seismic-discrimination-for-DiTing2.0

Introduction： We utilize the DiTing 2.0 dataset to develop a three-class network for identifying earthquakes, explosions, and collapses. The accuracies for discriminating earthquakes, explosions, and collapses using waveform and spectrogram datasets are 85% and 83%, respectively. The model trained on the DiTing 2.0 dataset, is successfully applied to regions in China.

'train_code8000_2.py': three-class discrimination network based on CNN. The input is waveform data, with the size of 1×8000, where 1 denotes the number of channels and 8000 signifies the number of sampling points in the seismic time domain. 
'diting_processing_collapse.py,diting_processing_earthquake.py,and diting_processing_explosion.py' is the processing script.
The training model (187 MB) is accessible on the Jianguoyun via https://www.jianguoyun.com/p/DUFFdaIQov3zBxjfpNUFIAA
Any questions, please contact: tianx@ecut.edu.cn
