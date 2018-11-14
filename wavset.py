import torch.utils.data.dataset
import torch
import numpy as np
import librosa
import os
import audioread

class WavSet(torch.utils.data.Dataset):
    def __init__(self, paths:list, srs:list, transform="stft", window_size=512):
        super(self.__class__, self).__init__()
        assert len(paths) != 0
        self.paths = paths
        self.srs = srs
        self.transform = transform
        self.window_size = window_size
        self.ys = self.load_waves()
        self.num_windows = [(len(y) // self.window_size) + 1 for y in self.ys]
        self.index_calculator = [sum(self.num_windows[:i+1]) for i in range(len(self.ys))]
        self.offset = 0

    def load_waves(self):
        import tqdm
        tqdm_range = tqdm.tqdm(range(len(self.paths)))
        ys = []
        for i in tqdm_range:
            path = self.paths[i]
            tqdm_range.set_description(f"Loading {os.path.basename(path)}...", refresh=True)
            sr = self.srs[i]
            try:
                y, sr = librosa.load(path=path, sr=sr)
            except audioread.NoBackendError:
                y = np.load(path)
            ys.append(y)
        return ys

    def __repr__(self):
        s = "<WavSet>\n"
        s += f"num_y {len(self.ys)}\n"
        s += f"num_windows {self.num_windows}\n"
        s += f"index calculator {self.index_calculator}"
        return s

    def __len__(self):
        return sum(self.num_windows)

    def __getitem__(self, index):
        ys_index = 0
        for cal in self.index_calculator:
            if index < cal:
                break
            ys_index += 1

        if ys_index > 0:
            window_index = index - self.index_calculator[ys_index - 1]
        else:
            window_index = index

        y = self.ys[ys_index]

        y_from = window_index * self.window_size
        y_to = min(len(y), y_from + self.window_size)
        return_y = y[y_from : y_to]
        if y_to - y_from < self.window_size:
            return_y = np.pad(return_y, (0, self.window_size - len(return_y)), mode='constant')

        # return_y = librosa.stft(return_y)
        # librosa.cqt()
        return return_y

    def save_cache(self, cache_dir="./cache/"):
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        for i in range(len(self.paths)):
            dest_dir = os.path.join(cache_dir, os.path.splitext(os.path.basename(self.paths[i]))[0])
            np.save(dest_dir, self.ys[i])

if __name__ == "__main__":
    # wavset = WavSet(["./data/ukulele/Kalei Gamiao-04-Kiss From A Rose-320k.mp3"], srs=[22500])
    wavset = WavSet(["./cache/Kalei Gamiao-04-Kiss From A Rose-320k.npy"], srs=[22500])

    import pdb
    pdb.set_trace()
    wavset[0]
    1/1