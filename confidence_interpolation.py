import numpy as np
import pandas as pd
from glob import glob
import re
import os




file = np.load("feature_archive/all_feature_interp951.npz", allow_pickle=True)
X = file["X"]
Y = file["Y"]
df = pd.read_csv("feature_archive/label_df_951.csv")
video_name = list(df.iloc[:,0])
all_feature_uninterp_path = [y for y in glob(os.path.join("D:/CU Files/IoT/Featurized_dataset/", "*.mp4.npz"))]
all_feature_name = [re.findall("Featurized_dataset\\\\(.+).mp4.npz", path)[0] for path in all_feature_uninterp_path]

# match all samples to their traj file
all_uninterp_path = []
for name in video_name:
    for path in all_feature_uninterp_path:
        if name in path:
            tmp = path.split("_")
            if tmp[-1][1:3] == " 2":
                continue
            else:
                all_uninterp_path.append(path)

def extract_scores(keypoints):
    scores = []
    for _, k in keypoints:
        if len(k) != 0:
            score = k[0, 3, :]
            scores.append(score)
    scores = np.stack(scores, axis=0)
    return scores


def score_interp(scores, n_frames=100):
    n_samples = scores.shape[0]
    if n_samples == 0:
        raise ValueError("Trajectories of length 0!")
    result = np.empty((n_frames, 17))
    scores = np.asarray(scores)
    dest_x = np.linspace(0, 100, n_frames)
    src_x = np.linspace(0, 100, n_samples)
    for j in range(17):
        result[:, j] = np.interp(
            dest_x,
            src_x,
            scores[:, j]
        )
    return result

all_scores_by_frame = np.empty((len(Y), 100, 17))   # n_samples, n_keypoints, n_frames
for i, path in enumerate(all_uninterp_path):
    traj = np.load(path, allow_pickle=True)
    scores_tmp = extract_scores(traj["keypoints"])
    score_by_frame = score_interp(scores_tmp)
    all_scores_by_frame[i,:,:] = score_by_frame

np.save("feature_archive/confidence_scores_by_frame.npy", all_scores_by_frame)