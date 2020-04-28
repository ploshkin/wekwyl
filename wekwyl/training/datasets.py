import bisect
import json
import os

import numpy as np
import torch as th
import torch.utils.data as data


def _list_files(folder):
    files = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file))
    ]
    return list(sorted(files))


def collate(batch):
    return {
        'frame': th.cat(
            [item['frame'].unsqueeze(dim=0) for item in batch],
            dim=0,
        ),
        'saliency': th.cat(
            [item['saliency'].unsqueeze(dim=0) for item in batch],
            dim=0,
        ),
        'fixations': [item['fixations'] for item in batch]
    }


class SaliencyDataset(data.dataset.Dataset):
    
    def __init__(
            self,
            videos,
            dataset_dir,
            frames_folder,
            maps_folder,
            fixations_filename,
            transform=None,
    ):
        super(SaliencyDataset, self).__init__()
        self.videos = videos
        self.dataset_dir = dataset_dir
        self.frames_dir = os.path.join(dataset_dir, frames_folder)
        self.maps_dir = os.path.join(dataset_dir, maps_folder)
        self.fixations_file = os.path.join(dataset_dir, fixations_filename)
        self.transform = transform

        self.frames = {
            video: _list_files(os.path.join(self.frames_dir, video))
            for video in videos
        }
        self.maps = {
            video: _list_files(os.path.join(self.maps_dir, video))
            for video in videos
        }
        with open(self.fixations_file) as ifile:
            fixations = json.load(ifile)
            self.fixations = {
                video: np.array(fixations[video])
                for video in self.videos
            }

        self.cumulative_length = np.cumsum(list(map(len, self.frames.values())))
        

    def __getitem__(self, index):
        video_index = bisect.bisect_right(self.cumulative_length, index)
        frame_index = (
            index
            if video_index == 0 
            else index - self.cumulative_length[video_index - 1]
        )
        video = self.videos[video_index]

        item =  {
            'frame': np.load(self.frames[video][frame_index]), 
            'saliency': np.load(self.maps[video][frame_index])[..., np.newaxis],
            'fixations': self.fixations[video][frame_index],
        }

        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self):
        return self.cumulative_length[-1]
