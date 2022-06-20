from glob import glob
import os
import re
import cv2
import numpy as np
import torch
from torchvision import models, transforms

import torch.nn as nn
import matplotlib.pyplot as plt

VIDEO_PATH = "D:/CU Files/IoT/IotLabData/data-clean/refrigerator/"
FEATURE_PATH = "D:/CU Files/IoT/Featurized_dataset/"
SAVE_PATH = "D:/CU Files/IoT/image_feature/resnet50/"


def get_path_dict(video_path="D:/CU Files/IoT/IotLabData/data-clean/refrigerator/",
                  feature_path="D:/CU Files/IoT/Featurized_dataset/"):
    all_video_path = [y for x in os.walk(video_path) for y in glob(os.path.join(x[0], '*.mp4'))]
    all_video_name = [re.findall("\\\\(\d)\\\\(.+).mp4", path)[0][1] for path in all_video_path]
    all_feature_path = [y for y in glob(os.path.join(feature_path, "*.mp4.npz"))]
    all_feature_name = [re.findall("Featurized_dataset\\\\(.+).mp4.npz", path)[0] for path in all_feature_path]

    print("{} of video files".format(len(all_video_name)))
    print("{} of feature files".format(len(all_feature_name)))
    print(len(list(set(all_video_name).intersection(set(all_feature_name)))))

    vf_path_dict = {}  # get a dictionary which records the video path and feature path of each sample
    for i, feature_name in enumerate(all_feature_name):
        try:
            idx = all_video_name.index(feature_name)
        except:
            continue
        feature_path = all_feature_path[i]
        video_path = all_video_path[idx]
        vf_path_dict[feature_name] = (video_path, feature_path)

    return vf_path_dict


def extract_feature(backbone,
                    video_path,
                    feature_path,
                    normalize={"mean": np.array([0.485, 0.456, 0.406]), "std": np.array([0.229, 0.224, 0.225])}):
    feature = np.load(feature_path, allow_pickle=True)
    boxes = feature["boxes"]

    cap = cv2.VideoCapture(video_path)
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert len(boxes) == video_length, "video length doesn't match"

    stacked_features = []
    for _, b in boxes:
        ret, frame = cap.read()
        if len(b) != 0:
            # get corner index of the box
            (w1, h1, w2, h2) = b[0][:4].astype('int')

            # # make sure the box is at least 7x7
            # if (w2 - w1) < 7:
            #     gap = 7 - (w2 - w1)
            #     if w2 + gap < 1279:
            #         w2 = w2 + gap
            #     else:
            #         w1 = w1 - gap
            # if (h2 - h1) < 7:
            #     gap = 7 - (h2 - h1)
            #     if h2 + gap < 719:
            #         h2 = h2 + gap
            #     else:
            #         h1 = h1 - gap
            image_box = frame[h1:h2, w1:w2, :]

            # normalize the image
            image_box = (image_box - normalize["mean"]) / normalize["std"]
            image_box = image_box.transpose((2, 1, 0))
            image_box_tensor = torch.Tensor(image_box).unsqueeze(0)

            # extract feature
            backbone.eval()
            with torch.no_grad():
                features = backbone(image_box_tensor).numpy()
                stacked_features.append(features)

    stacked_features = np.vstack(stacked_features)

    return stacked_features


if __name__ == "__main__":
    path_dict = get_path_dict(VIDEO_PATH, FEATURE_PATH)

    # load backbone network and remove classifier layer
    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()

    failure_list = []
    for video_name, (video_path, feature_path) in path_dict.items():
        file_name = SAVE_PATH + video_name + ".mp4.npz"
        if os.path.exists(file_name):
            print(file_name, "saved")
            continue
        print(file_name, end=" ")

        try:
            feature = extract_feature(backbone, video_path, feature_path)
            np.savez(file_name, feature=feature)
            print("saved")
        except:
            failure_list.append(video_name)
            print("failed")

    with open(r"failure_list.txt", "w") as fp:
        for item in failure_list:
            fp.write("%s\n" % item)

    print("{} of videos failed".format(len(failure_list)))
