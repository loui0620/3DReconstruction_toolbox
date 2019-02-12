import numpy as np

def read_matrix(path, astype=np.float64):
    with open(path, 'r') as f:
        arr = []
        for line in f:
            arr.append([(token if token != '*' else -1)
                        for token in line.strip().split()])
        return np.asarray(arr).astype(astype)

def cart2homo(arr):
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))

def homo2cart(arr):
    # arr has shape: dimensions x num_points
    num_rows = len(arr)
    if num_rows == 1 or arr.ndim == 1:
        return arr

    return np.asarray(arr[:num_rows - 1] / arr[num_rows - 1])