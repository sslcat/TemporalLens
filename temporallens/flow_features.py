"""Pilot Farneback descriptors shared by cached training and video inference."""

import cv2
import numpy as np


def flow_vector(previous, current):
    field = cv2.calcOpticalFlowFarneback(
        previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    border = np.concatenate(
        (
            field[:3].reshape(-1, 2),
            field[-3:].reshape(-1, 2),
            field[:, :3].reshape(-1, 2),
            field[:, -3:].reshape(-1, 2),
        )
    )
    field -= np.median(border, axis=0)
    u, v = field.transpose(2, 0, 1)
    magnitude = np.hypot(u, v)
    angle = ((np.arctan2(v, u) + np.pi) * 4 / np.pi).astype(int) % 8
    yy, xx = np.indices(u.shape)
    mass = magnitude.sum() + 1e-6
    dx = xx - (xx * magnitude).sum() / mass
    dy = yy - (yy * magnitude).sum() / mass
    radial = (dx * u + dy * v) / np.maximum(1, np.hypot(dx, dy))
    values = []
    for rows in np.array_split(np.arange(u.shape[0]), 3):
        for cols in np.array_split(np.arange(u.shape[1]), 4):
            cell = np.ix_(rows, cols)
            values.extend(
                [
                    u[cell].mean(),
                    v[cell].mean(),
                    radial[cell].mean(),
                    magnitude[cell].mean(),
                ]
            )
            values.extend(
                np.bincount(
                    angle[cell].ravel(), weights=magnitude[cell].ravel(), minlength=8
                )
                / len(rows)
                / len(cols)
            )
    return np.asarray(values, np.float32)


def gray(rgb):
    return cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (80, 60), interpolation=cv2.INTER_AREA
    )


def flow_sequence(images):
    return np.stack(
        [flow_vector(images[max(i - 1, 0)], im) for i, im in enumerate(images)]
    )
