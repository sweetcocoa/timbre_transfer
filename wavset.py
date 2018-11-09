import torch.utils.data.dataset
import torch
import numpy as np
import librosa
import os

class WavSet(torch.utils.data.dataset):
    def __init__(self, paths:list, srs:list, transform="stft", window_size=512):
        super(self.__class__, self).__init__()
        self.paths = paths
        self.srs = srs
        self.transform = transform
        self.window_size = window_size
        self.ys = self.load_waves()
        self.num_windows = [(len(y) // self.window_size) + 1 for y in self.ys]
        self.offset = 0

    def load_waves(self):
        import tqdm
        tqdm_range = tqdm.tqdm(range(len(self.paths)))
        ys = []
        for i in tqdm_range:
            path = self.paths[i]
            tqdm_range.set_description(f"Loading {os.path.basename(path)}...", refresh=True)
            sr = self.srs[i]
            y, sr = librosa.load(path=path, sr=sr)
            ys.append(y)
        return ys

    def __len__(self):
        return sum(self.num_windows)

    def __getitem__(self, index):
        pass
