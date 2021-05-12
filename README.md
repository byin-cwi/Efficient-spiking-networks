# Adaptive (SRNN) Spiking Recurrent  Neural network 

This code implements the adaptive spiking recurrent network with learnable parameters on [Pytorch]([PyTorch](https://pytorch.org/)) for various tasks. 
This is scientific software, and as such subject to many modifications; we aim to further improve the software to become more user-friendly and extendible in the future. 


For those wanting to try it out: The best place to start is the Jupyter notebook. It’s a quick demonstration of spiking neural networks.
This is what you need:
1) A working version of python and Pytorch This should be easy: either use the Google Colab facilities, or do a simple installation on your laptop could probabily using pip. ([Start Locally | PyTorch](https://pytorch.org/get-started/locally/))
2) The datasets: 
	1. ECG: The original QTDB dataset was downloaded from [PhysioNet](https://physionet.org/content/qtdb/1.0.0/). The encoded dataset is attached in /data/ folder 
  	2. (P)S-MNIST: This dataset can easily be found in torchvision.datasets.MNIST
 	3. SHD and SSC: the Spiking Heidelberg dataset can be downloaded from the [official sit]( https://compneuro.net/datasets/)
	4. SoLi dataset: The official   [dataset](https://polybox.ethz.ch/index.php/s/wG93iTUdvRU8EaT) can be easily downloaded and other necessary information is well documented at the official git[GitHub - simonwsw/deep-soli: Gesture Recognition Using Neural Networks with Google’s Project Soli Sensor](https://github.com/simonwsw/deep-soli)
	5. Google [Speech Commands Dataset](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz) 
	6. TIMIT Acoustic-Phonetic Continuous Speech Corpus can be found here: LDC (https://catalog.ldc.upenn.edu/LDC93S1)

3) Pre-requisites 

- Python 3: HDF5, OpenCV 2 interfaces for python.
- Pytorch>=1.5.1
- Preprocessing packages: [librosa]([librosa — librosa 0.8.0 documentation](https://librosa.org/doc/latest/index.html)),[tdqm]([GitHub - tqdm/tqdm: A Fast, Extensible Progress Bar for Python and CLI](https://github.com/tqdm/tqdm)),[tables]([tables · PyPI](https://pypi.org/project/tables/)),[wfdb]([wfdb — wfdb 3.3.0 documentation](https://wfdb.readthedocs.io/en/latest/)),[klepto]([klepto · PyPI](https://pypi.org/project/klepto/)) and [Scipy]([SciPy.org — SciPy.org](https://www.scipy.org/index.html)),[Jupyter Notebook]([Project Jupyter | Home](https://jupyter.org/))
- Optional: SSC and SHD dataset require more than 32GB RAM to hold the dataset. Training on GPU may accelerate the evaluation.

4) Code running
* Data preprocessing. The datasets(ECG,SHD,SSC,SoLi,TIMIT) are required to arrange the dataset for further training. 
‘’’
mkdir  data
python generate_dataset.py
‘’’
Note that you can also change the sampling frequency to generate a higher precision dataset. 
* Model training. 
‘’’
mkdir  model
python train.py
‘’’
* Visualization of the results(Demo). The results can be visualized by [Jupyter Notebook]([Project Jupyter | Home](https://jupyter.org/))
* Pre-trained models are attached in the folder of each dataset. Because the training takes a quite long time and 


Finally, we’d love to hear from you if you have any comments or suggestions.

### References


[1]. Bojian Yin, Federico Corradi, Sander M. Bohté. **Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks**

[2]. Bojian Yin, Federico Corradi, Sander M. Bohté. **Effective and efficient computation with multiple-timescale spiking recurrent neural networks**


