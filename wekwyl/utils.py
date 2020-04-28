import dataclasses

import cv2
import numpy as np
from tqdm import tqdm


@dataclasses.dataclass
class Point:
    x: float
    y: float

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __truediv__(self, number):
        if isinstance(number, (int, float)):
            return Point(self.x / number, self.y / number)
        else:
            raise TypeError

    def __floordiv__(self, number):
        if isinstance(number, (int, float)):
            return Point(self.x // number, self.y // number)
        else:
            raise TypeError

    def __mul__(self, number):
        if isinstance(number, (int, float)):
            return Point(self.x * number, self.y * number)
        else:
            raise TypeError

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


def read_video(path, size=None, tqdm=None):
    assert size is None or len(size) == 2

    cap = cv2.VideoCapture(path)

    n, h, w = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    size = size if size != (w, h) else None
    shape = (n, size[1], size[0], 3) if size else (n, h, w, 3)
    frames = np.zeros(shape, dtype=np.uint8)

    frame_iterator = range(n)
    if tqdm:
        frame_iterator = tqdm(frame_iterator)

    for index in frame_iterator:
        ret, frame = cap.read()
        if not ret:
            break

        if size:
            if size[0] > w or size[1] > h:
                interpolation = cv2.INTER_LANCZOS4
            else:
                interpolation = cv2.INTER_AREA

            frame = cv2.resize(
                frame, size, interpolation=interpolation,
            )

        frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()
    return frames
