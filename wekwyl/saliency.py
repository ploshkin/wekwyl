from collections import defaultdict
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import scipy as sp
from tqdm import tqdm

from utils import Point


def _update_fixations(df, fixations, fps):
    viewer_fixations = defaultdict(list)
    start = df['RecordingTime [ms]'].min()
    for index, item in df.iterrows():
        frame = int(np.floor((item['RecordingTime [ms]'] - start) * fps / 1000))
        if item['Category Binocular'] == 'Visual Intake':
            x = item['Point of Regard Binocular X [px]']
            y = item['Point of Regard Binocular Y [px]']
            viewer_fixations[frame].append(Point(x, y))

    for frame, points in viewer_fixations.items():
        # Get first fixation at the frame.
        fixations[frame].append(points[0])


def get_fixations(file_paths, fps):
    fixations = defaultdict(list)
    for file in tqdm(file_paths):
        try:
            df = pd.read_csv(file)
        except:
            continue
        _update_fixations(df, fixations, fps)

    return fixations


def _normalize(arr) -> np.ndarray:
    return (arr - arr.min()) / arr.max()


def get_equatorial_center_prior(w, h, sigma=1.0):
    bell = np.exp(-((np.arange(h) - h / 2) / sigma) ** 2)
    cp = np.repeat(bell[..., np.newaxis], w, axis=1)
    return _normalize(cp)


def make_saliency_map(image, points):
    h, w = image.shape[:2]

    fixation_map = np.zeros((h, w))
    for pt in points:
        y, x = round(pt.y), round(pt.x)
        fixation_map[y, x] = 1

    kernel = (w // 10 + 1, w // 10 + 1)
    sigma = w // 50
    saliency_map = cv2.GaussianBlur(fixation_map, kernel, sigma)
    return _normalize(saliency_map)


def make_heatmap(image, saliency_map):
    inverted = 1 - saliency_map
    discrete = np.rint(inverted * 255).astype(np.uint8)
    rgb = np.repeat(discrete[..., np.newaxis], 3, axis=2)
    heatmap = cv2.applyColorMap(rgb, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)


def draw_fixations(image, points):
    h, w = image.shape[:2]
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    r = round(w * 0.007)
    color = (255, 0, 0)

    for pt in points:
        bbox = [pt.x - r, pt.y - r, pt.x + r, pt.y + r]
        draw.ellipse(bbox, fill=color)

    return img


def make_vizualization(image, points, kind='heatmap'):
    if kind == 'points':
        img = draw_fixations(image, points)

    elif kind == 'heatmap':
        saliency = make_saliency_map(image, points)
        heatmap = make_heatmap(image, saliency)
        img = Image.fromarray(heatmap)

    else:
        raise RuntimeError(f'Unknown kind: `{kind}`')
    
    return img
