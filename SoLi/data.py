import os
import numpy as np
import scipy.io.wavfile as wav
import torch
from torch.utils.data import Dataset
from utils import txt2list, split_wav
import librosa
from sklearn.preprocessing import normalize


class SpeechCommandsDataset(Dataset):
    def __init__(self, data_root, label_dct, mode, transform=None, max_nb_per_class=None):
        
        assert mode in ["train", "valid", "test"], 'mode should be "train", "valid" or "test"' 
        
        self.filenames = []
        self.labels = []
        self.mode = mode
        self.transform = transform
        
        
        if self.mode == "train" or self.mode == "valid":
            testing_list = txt2list(os.path.join(data_root, "testing_list.txt"))
            validation_list = txt2list(os.path.join(data_root, "validation_list.txt"))
            validation_list += txt2list(os.path.join(data_root, "silence_validation_list.txt"))
        else:
            testing_list = []
            validation_list = []
        
        
        for root, dirs, files in os.walk(data_root):
            if "_background_noise_" in root:
                continue
            for filename in files:
                if not filename.endswith('.wav'):
                    continue
                command = root.split("/")[-1]
                label = label_dct.get(command)
                if label is None:
                    print("ignored command: %s"%command)
                    break
                partial_path = '/'.join([command, filename])
                
                testing_file = (partial_path in testing_list)
                validation_file = (partial_path in validation_list)
                training_file = not testing_file and not validation_file
                
                if (self.mode == "test") or (self.mode=="train" and training_file) or (self.mode=="valid" and validation_file):
                    full_name = os.path.join(root, filename)
                    self.filenames.append(full_name)
                    self.labels.append(label)
                
        if max_nb_per_class is not None:
            
            selected_idx = []
            for label in np.unique(self.labels):
                label_idx = [i for i,x in enumerate(self.labels) if x==label]
                if len(label_idx) < max_nb_per_class:
                    selected_idx += label_idx
                else:
                    selected_idx += list(np.random.choice(label_idx, max_nb_per_class))
            
            self.filenames = [self.filenames[idx] for idx in selected_idx]
            self.labels = [self.labels[idx] for idx in selected_idx]
        
                
        if self.mode == "train":
            label_weights = 1./np.unique(self.labels, return_counts=True)[1]
            label_weights /=  np.sum(label_weights)
            self.weights = torch.DoubleTensor([label_weights[label] for label in self.labels])
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        filename = self.filenames[idx]
        item = wav.read(filename)[1].astype(float)
        m = np.max(np.abs(item))
        if m > 0:
            item /= m
        if self.transform is not None:
            item = self.transform(item)
            
        label = self.labels[idx]
        
        return item, label
    
class Pad:

    def __init__(self, size):
        
        self.size = size
        
    def __call__(self, wav):
        wav_size = wav.shape[0]
        pad_size = (self.size - wav_size)//2
        padded_wav = np.pad(wav, ((pad_size, self.size-wav_size-pad_size),), 'constant', constant_values=(0, 0))
        return padded_wav
    
    
class RandomNoise:
    
    
    def __init__(self, noise_files, size, coef):
        
        
        self.size = size
        self.noise_files = noise_files
        self.coef = coef
        
        
    def __call__(self, wav):
        
        
        if np.random.random() < 0.8:
            
            noise_wav = get_random_noise(self.noise_files, self.size)
            noise_power = (noise_wav**2).mean()
            sig_power = (wav**2).mean()
            
            noisy_wav = wav + self.coef  * noise_wav * np.sqrt(sig_power / noise_power) 
            
        else:
            
            noisy_wav = wav
            
        return noisy_wav
    
    
class RandomShift:
    
    def __init__(self, min_shift, max_shift):
        
        self.min_shift = min_shift
        self.max_shift = max_shift
        
    def __call__(self, wav):
        

        shift = np.random.randint(self.min_shift, self.max_shift+1)
        shifted_wav = np.roll(wav, shift)
    
        if shift > 0:
            shifted_wav[:shift] = 0
        elif shift < 0:
            shifted_wav[shift:] = 0
        
        return shifted_wav
    
    
class MelSpectrogram:
    
    def __init__(self, sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order=None, stack=True):
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.delta_order = delta_order
        self.stack=stack
        
        
    def __call__(self, wav):
        
        S = librosa.feature.melspectrogram(wav,
                           sr=self.sr,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           n_mels=self.n_mels, 
                           fmax=self.fmax,
                           fmin=self.fmin)
    
        M = np.max(np.abs(S))
        if M > 0:
            feat = np.log1p(S/M)
        else:
            feat = S
    
        if self.delta_order is not None and not self.stack:
            feat = librosa.feature.delta(feat, order=self.delta_order)
            return np.expand_dims(feat.T, 0)
        
        elif self.delta_order is not None and self.stack:
            
            feat_list = [feat.T]
            for k in range(1, self.delta_order+1):
                feat_list.append(librosa.feature.delta(feat, order=k).T)
            return np.stack(feat_list)
        
        else:
            return np.expand_dims(feat.T, 0)
    
    
class Rescale:
    
    def __call__(self, input):
        
        std = np.std(input, axis=1, keepdims=True)
        std[std==0]=1
        
        return input/std

class Normalize:
    
    def __call__(self, input):
        
        input_ = (input>0.1)*input
        std = np.std(input_, axis=1, keepdims=True)
        std[std==0]=1
        
        return input/std

    
class WhiteNoise:
    
    def __init__(self, size, coef_max):
        
        
        self.size = size
        self.coef_max = coef_max
        
        
    def __call__(self, wav):
            
        noise_wav = np.random.normal(size = self.size)
        noise_power = (noise_wav**2).mean()
        sig_power = (wav**2).mean()

        coef = np.random.uniform(0., self.coef_max)

        noisy_wav = wav + coef  * noise_wav * np.sqrt(sig_power / noise_power) 
            
        return noisy_wav