from typing import List
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


def read_video(
        path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
) -> List[np.ndarray]:
    frames = []
    cap = cv2.VideoCapture(path)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not start_frame:
        start_frame = 0
    start_frame = min(max(0, start_frame), num_frames - 1)

    if not end_frame:
        end_frame = num_frames - 1
    end_frame = min(max(0, end_frame), num_frames - 1)

    for index in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= index <= end_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames