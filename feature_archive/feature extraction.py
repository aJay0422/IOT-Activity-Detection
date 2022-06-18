from glob import glob
import os
import re
import cv2
import numpy as np

VIDEO_PATH = "D:/CU Files/IoT/IotLabData/data-clean/refrigerator/"
FEATURE_PATH = "D:/CU Files/IoT/Featurized_dataset/"


def get_path_dict(video_path="D:/CU Files/IoT/IotLabData/data-clean/refrigerator/",
                  feature_path="D:/CU Files/IoT/Featurized_dataset/"):
    all_video_path = [y for x in os.walk(video_path) for y in glob(os.path.join(x[0], '*.mp4'))]
    all_video_name = [re.findall("\\\\(\d)\\\\(.+).mp4", path)[0][1] for path in all_video_path]
    all_feature_path = [y for y in glob(os.path.join(feature_path, "*.mp4.npz"))]
    all_feature_name = [re.findall("Featurized_dataset\\\\(.+).mp4.npz", path)[0] for path in all_feature_path]

    print("{} of video files".format(len(all_video_name)))
    print("{} of feature files".format(len(all_feature_name)))
    print(len(list(set(all_video_name).intersection(set(all_feature_name)))))

    vf_path_dict = {}
    for i, feature_name in enumerate(all_feature_name):
        try:
            idx = all_video_name.index(feature_name)
        except:
            continue
        feature_path = all_feature_path[i]
        video_path = all_video_path[idx]
        vf_path_dict[feature_name] = (video_path, feature_path)

    return vf_path_dict

def extract_feature(backbone, video_path, feature_path):
    feature = np.load(feature_path, allow_pickle=True)
    boxes = feature["boxes"]

    cap = cv2.VideoCapture(video_path)
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert len(boxes) == video_length, "video length doesn't match"

    for _, b in boxes:
        ret, frame = cap.read()
        if len(b) != 0:
            (w1, h1, w2, h2) = b[0][:4]

            print(frame.shape)
            break


if __name__ == "__main__":
    path_dict = get_path_dict(VIDEO_PATH, FEATURE_PATH)
    for video_name, (video_path, feature_path) in path_dict.items():
        extract_feature(None, video_path, feature_path)

        break