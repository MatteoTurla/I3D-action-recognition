import torch
from torch.utils import data

import os
import glob
import pdb
import json
import queue

from random import randint

class Dataset(data.Dataset):
    def __init__(self, root_dataset, n_frames, campionamento=1, balance=True, padding=True, temporal_annotation=None):
        #if padding False remove file with less than n_frames
        #if balance True balance dataset
        self.root_dataset = root_dataset

        self.list_videos = []
        self.list_labels = []

        self.n_frames = n_frames
        self.campionamento = campionamento #stride frame consecutivi
        self.balance = balance
        self.padding = padding
        self.removed = 0
        self.temporal_annotation = temporal_annotation

        self.classes, self.class_to_idx, self.idx_to_class = self._find_classes(self.root_dataset)

        #crea lista video e label leggendo dalla directory root
        for target in self.class_to_idx.keys():
            d = os.path.join(self.root_dataset, target)
            files = glob.glob(os.path.join(d, "*.pt"))
            for file in files:
                self.list_videos.append(file)
                self.list_labels.append(self.class_to_idx[target])

        if not padding:
            self.list_videos, self.list_labels = self._removeUnfeasible()    

        if balance:
            self._balance()
        
        if not temporal_annotation == None: 
            print('loading temporal annotation')
            with open(temporal_annotation, 'r') as f:
                self.crop_temp = json.load(f)

    def _find_classes(self, _dir):
        classes = [d.name for d in os.scandir(_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {i: classes[i] for i in range(len(classes))}
        return classes, class_to_idx, idx_to_class

    def _balance(self):
        print('Dataset -> balancing')
        bins = self.bincount()
        _max = bins.max().item()
        for i in range(len(bins)):
            _bin = bins[i].item()
            replicate = _max // _bin - 1
            _random = _max % _bin

            if i > 0:
                    start = bins[:i].sum().item()
            else: 
                start = 0

            for rep in range(replicate):
                for k in range(_bin):
                    self.list_videos.append(self.list_videos[k+start])
                    self.list_labels.append(self.list_labels[k+start])
            for _rand in range(_random):
                j = randint(start, start+_bin - 1)
                self.list_videos.append(self.list_videos[j])
                self.list_labels.append(self.list_labels[j])

        #sanity check
        bins = self.bincount()
        for i in range(1, len(bins)):
            if bins[i].item() != bins[i-1].item():
                raise Exception('Balancing gone wrong!')
        print('Done..')

    def _removeUnfeasible(self):
        print('Dataset -> searching for unfeasible')
        toDel = queue.Queue()
        for i in range(len(self.list_videos)):
            video = torch.load(self.list_videos[i])
            if video.shape[1] < self.n_frames:
                toDel.put(i)
        print('Dataset -> deleting unfeasible')
        list_videos_tmp = []
        list_labels_tmp = []
        print(toDel.qsize())
        todel = -1
        if not toDel.empty():
            todel = toDel.get()
        for i in range(len(self.list_videos)):
            if i == todel:
                self.removed += 1
                if not toDel.empty():
                    todel = toDel.get()
            else:
                list_videos_tmp.append(self.list_videos[i])
                list_labels_tmp.append(self.list_labels[i])
        print('Done..')
        return list_videos_tmp, list_labels_tmp

    def _cropTemp(self, video, video_name, label):
        _key = self.idx_to_class[label]+"_"+video_name
        proposal = self.crop_temp[_key]['proposal']
        if isinstance(proposal, str):
            print('no annotations')
            return video
        else:
            return video[:,proposal[0]:proposal[1]+1,:,:]

    def _campionamento(self, video, downsample):
        depth = video.shape[1]
        if(downsample == 1):
            return video
        if(depth // self.n_frames >= downsample):
            return video[:,::downsample,:,:]
        else:
            return self._campionamento(video, downsample -1)

    def _center(self, video):
        center = video.shape[1] // 2
        frame = self.n_frames // 2
        return video[:,center-frame:center+frame,:,:]

    def auto_pading(self, data):
        size = self.n_frames
        C, T, H, W = data.shape
        if T < size:
            begin = (size - T) // 2
            data_padded = torch.zeros((C, size, H, W))
            data_padded[:, begin:begin + T, :, :] = data
            return data_padded
        else:
            return data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        video = self.list_videos[index]
        video_name = video.split('.')[-2].split('/')[-1]
        X = torch.load(video)
        y = self.list_labels[index]
        if self.padding:
            X = self.auto_pading(X)
        if not self.temporal_annotation == None:
            X = self._cropTemp(X, video_name, y)
        X = self._campionamento(X, self.campionamento)
        X = self._center(X)

        return X, y

    def bincount(self):
        return torch.bincount(torch.tensor(self.list_labels))

    def print(self):
        print("Dataset:", self.root_dataset)
        print("Classes:", self.classes)
        print("Classes to index:")
        for c in self.class_to_idx:
            print("Label:", c, "index:", self.class_to_idx[c])
        print("Numero di frame:", self.n_frames,)
        print("Downsample:", self.campionamento)
        print("Balance:", self.balance)
        print("Padding:", self.padding)
        print("removed:", self.removed)
        print("Temporal annotation:", self.temporal_annotation),
        print("Numero Azioni", self.__len__())
        bins = self.bincount()
        for idx, _bin in enumerate(bins):
            print(self.idx_to_class[idx], "\t", _bin.item())

if __name__ == '__main__':
    dataset_test_path = '/home/Dataset/aggregorio_videos_pytorch_boxcrop/daily_life'+'/test'
    dataset_test = Dataset(dataset_test_path, 32, 2, balance=False, padding=False, temporal_annotation='/home/Dataset/crop_temp.json')
    loader_test = data.DataLoader(
                            dataset_test,
                            batch_size=32,
                        	shuffle=False,
                        	pin_memory=True,
                            num_workers = 4
                        )
    for X, y in loader_test:
        print(X.shape)
