import numpy as np
import cv2
import pandas as pd
import torch
from glob import glob
import re
import os


from Transformer.model import transformer_huge
from Transformer.utils import prepare_data

LABEL_DECODER = {0:"no_interaction",
                 1:"open_close_fridge",
                 2:"put_back_item",
                 3:"screen_interaction",
                 4:"take_out_item"}

def add_keypoints(img, keypoints):
    """
    add 17 keypoints to a frame
    :param img: the frame to add keypoints
    :param keypoints: a vector of length 34
    :return: a frame
    """
    for i in range(17):
        coord = (int(keypoints[i]), int(keypoints[i+17]))
        img = cv2.circle(img, center=coord, radius=2, color=(0, 0, 0), thickness=cv2.FILLED)
        # img = cv2.putText(img, str(i), coord, cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=2)


    # add left arm
    index = [5, 7, 9]
    for i in range(len(index)-1):

        img = cv2.line(img, (int(keypoints[index[i]]), int(keypoints[index[i]+17])),
                       (int(keypoints[index[i+1]]), int(keypoints[index[i+1]+17])), color=(0, 0, 0))

    # add right arm
    index = [6, 8, 10]
    for i in range(len(index) - 1):
        img = cv2.line(img, (int(keypoints[index[i]]), int(keypoints[index[i] + 17])),
                       (int(keypoints[index[i + 1]]), int(keypoints[index[i + 1] + 17])), color=(0, 0, 0))


    return img


def extract_feature(traj):
    video_length = len(traj["keypoints"])
    keypoints_uninterp = np.zeros((video_length, 34))
    for i in range(video_length):
        features = traj["keypoints"][i][1]
        if len(features) == 0:
            continue
        keypoints_uninterp[i, :] = features[0][:2, :].reshape(34)

    if traj["metadata"].item()["w"] == 1920:
        keypoints_uninterp /= 1.5

    return keypoints_uninterp.transpose(1, 0)





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = transformer_huge()
model.to(device)
model.load_state_dict(torch.load("Transformer/experiment/Transformer_huge_1.pth", map_location=device))

file = np.load("feature_archive/all_feature_interp951.npz", allow_pickle=True)
X = file["X"].reshape(-1, 100, 34).transpose(0, 2, 1)
Y = file["Y"]
df = pd.read_csv("feature_archive/label_df_951.csv")
video_name = list(df.iloc[:,0])
all_feature_uninterp_path = [y for y in glob(os.path.join("D:/CU Files/IoT/Featurized_dataset/", "*.mp4.npz"))]
all_feature_name = [re.findall("Featurized_dataset\\\\(.+).mp4.npz", path)[0] for path in all_feature_uninterp_path]
# fail_index = []
# with torch.no_grad():
#     model.eval()
#     for i in range(len(Y)):
#         X_tmp = torch.Tensor(X[i]).unsqueeze(0).to(device)
#         Y_tmp = Y[i]
#         logit = model(X_tmp)
#         Y_pred = torch.argmax(logit).cpu().squeeze().numpy()
#         if not Y_pred == Y_tmp:
#             fail_index.append(i)
#
# print(f"{len(fail_index)} failed.")
# print(fail_index)


fail_index = [107, 153, 247, 250, 270, 275, 358, 364, 371, 394, 443, 444, 578, 582, 642, 680, 698, 713, 717, 721, 748, 749, 751, 809, 821, 870, 899, 913, 914, 915]


# good examples: 0. Feature doesn't seem correct.
#                2. The man in the video did extra movement, explains the attention mechanism
#                5 16 17 18 19 23 24. We need to know if there is an object in the person's hands
#                10 11. The person is not in the video at all
#                12. Only half of the man in the video
#                14. Can not classify it even using the video
#                20. Did reach his hand into the fridge, but didn't take out or put back anything.
#                21. Depth is important
#                25 26. Didn't take out item inside fridge, but from the door
#                27. The original video has 2700 frames, and also the object problem
#                28. Keypoint estimation is not correct
#                29. Didn't do the "no interaction" in a common way


index = fail_index[16]
index = 192   # 192, 638, 1
single_video = X[index][:,:]
failed_name = video_name[index]
for path in all_feature_uninterp_path:
    if failed_name in path:
        print(path)
        traj = np.load(path, allow_pickle=True)
        break

single_video_uninterp = extract_feature(traj)


def visualize(video, interp=True):

    if interp:
        cls = LABEL_DECODER[Y[index]]
        with torch.no_grad():
            model.eval()
            logits = model(torch.Tensor(video.copy()).unsqueeze(0).to(device))
            y_pred = LABEL_DECODER[int(torch.argmax(logits).cpu().numpy())]
            weights = model.blocks[6].attn.attn[0, :, 0, 1:].cpu().numpy()
            weights = weights.mean(axis=0)
            weights = (weights - weights.min()) / (weights.max() - weights.min())


        i = 0
        while True:
            img = np.ones((720, 1280, 3))
            img = add_keypoints(img, video[:,i % 100])
            img = cv2.putText(img, "frame {}/100".format(i % 100 + 1), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            img = cv2.putText(img, "True: " + cls, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            img = cv2.putText(img, "Pred: " + y_pred, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            if weights[i%100] > 0.5:
                img = cv2.putText(img, "Attention: {:.0f}%".format(weights[i % 100] * 100), (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
            else:
                img = cv2.putText(img, "Attention: {:.0f}%".format(weights[i % 100] * 100), (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)

            cv2.imshow("Image", img)
            if weights[i%100] > 0.5:
                cv2.waitKey(200)
            else:
                cv2.waitKey(200)
            i += 1
    else:
        cls = LABEL_DECODER[Y[index]]
        video_length = video.shape[1]
        i = 0
        while True:
            img = np.ones((720, 1280))
            img = add_keypoints(img, video[:, i % video_length])
            img = cv2.putText(img, "frame {}/{}".format(i % video_length + 1, video_length), (20, 60), cv2.FONT_ITALIC, 1, (0, 0, 255),
                              thickness=2)
            img = cv2.putText(img, "True: " + cls, (20, 120), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=2)

            cv2.imshow("Image", img)
            cv2.waitKey(200)
            i += 1


visualize(single_video, interp=True)
# visualize(single_video_uninterp, interp=False)