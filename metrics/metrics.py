import torchaudio
from pystoi import stoi
import numpy as np
from pesq import pesq_batch

class PESQ:
    def __init__(self, in_sr=16000, sr=16000, on_error=1, mode='wb', n_processor=1):
        self.in_sr = in_sr
        self.sr = sr
        self.on_error = on_error
        self.mode = mode
        self.n_processor = n_processor
        if in_sr != sr:
            self.resampler = torchaudio.transforms.Resample(in_sr, sr)
        else:
            self.resampler = None
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        
    def update(self, x, y):
        if self.resampler:
            x = self.resampler(x.float().cpu())
            y = self.resampler(y.float().cpu())
        x = x.float().cpu().numpy()
        y = y.float().cpu().numpy()
        min_len = min(x.shape[1], y.shape[1])
        x = x[:,:min_len]
        y = y[:,:min_len]
        n = x.shape[0]
        val = np.mean(pesq_batch(fs=self.sr, ref=x, deg=y, on_error=self.on_error, mode=self.mode, n_processor=self.n_processor))
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        return self.avg



class STOI:
    def __init__(self, in_sr=16000, sr=16000):
        self.in_sr = in_sr
        self.sr = sr
        if in_sr != sr:
            self.resampler = torchaudio.transforms.Resample(in_sr, sr)
        else:
            self.resampler = None
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        
    def update(self, x, y):
        if self.resampler:
            x = self.resampler(x.float().cpu())
            y = self.resampler(y.float().cpu())
        x = x.float().cpu().numpy()
        y = y.float().cpu().numpy()
        min_len = min(x.shape[1], y.shape[1])
        x = x[:,:min_len]
        y = y[:,:min_len]
        n = x.shape[0]
        val = 0
        for ref, deg in zip(x,y):
            val += stoi(x=ref, y=deg, fs_sig=self.sr, extended=False) / n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        return self.avg