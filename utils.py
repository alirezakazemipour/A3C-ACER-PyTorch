import cv2
import numpy as np


def preprocessing(x):
    img = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    return img


def stack_states(stacked_frames, state, is_new_episode):
    frame = preprocessing(state)

    if is_new_episode:
        stacked_frames = np.stack([frame for _ in range(4)], axis=0)
    else:
        stacked_frames = stacked_frames[1:, ...]
        stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=0)], axis=0)
    return stacked_frames
