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
    batch = list(filter(lambda x: x is not None, batch))
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
            frac=None,
    ):
        super(SaliencyDataset, self).__init__()
        self.videos = videos
        self.dataset_dir = dataset_dir
        self.frames_dir = os.path.join(dataset_dir, frames_folder)
        self.maps_dir = os.path.join(dataset_dir, maps_folder)
        self.fixations_file = os.path.join(dataset_dir, fixations_filename)
        self.transform = transform
        self.frac = frac if frac else 1.0

        assert 0 <= self.frac <= 1

        with open(self.fixations_file) as ifile:
            fixations = json.load(ifile)

        self.frames = {}
        self.maps = {}
        self.fixations = {}

        for video in videos:
            frames = _list_files(os.path.join(self.frames_dir, video))
            maps = _list_files(os.path.join(self.maps_dir, video))
            fixs = fixations[video]

            assert len(frames) <= len(maps), f'Video: "{video}"'
            assert len(frames) <= len(fixs), f'Video: "{video}"'

            num_frames = len(frames)
            frac_frames = np.ceil(self.frac * num_frames).astype(np.int32)
            indices = sorted(np.random.choice(num_frames, frac_frames, replace=False))

            self.frames[video] = [frames[i] for i in indices]
            self.maps[video] = [maps[i] for i in indices]
            self.fixations[video] = [fixs[i] for i in indices]

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
