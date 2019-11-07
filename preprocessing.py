import numpy as np
from cv2 import COLOR_RGB2GRAY, cvtColor

def to_grayscale(prev_frames):
    result = []
    prev_frames = np.array(prev_frames)

    for i in range(len(prev_frames[:, :, 1])):
        result.append(cvtColor(prev_frames[i, :, :], COLOR_RGB2GRAY))
    result = np.array(result)
    print("RESULT", result.size)
    return result


def downsample(prev_frames):
    result = []
    for i in range(len(prev_frames)):
        result[i] = prev_frames[i][::2, ::2]
    return result


def preprocess(prev_frames):
    #new_frames = downsample(to_grayscale(prev_frames))
    new_frames = to_grayscale(prev_frames)
    new_frames = np.array(new_frames)
    print(new_frames)
    new_frames = [frame[:, :, np.newaxis] for frame in new_frames]
    return np.concatenate(new_frames, axis=2)