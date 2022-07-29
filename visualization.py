import numpy as np
import cv2


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
        img = cv2.circle(img, center=coord, radius=2, color=(0, 0, 255), thickness=cv2.FILLED)

    return img


file = np.load("feature_archive/all_feature_interp951.npz")
X = file["X"].reshape(-1, 100, 34).transpose(0, 2, 1)
Y = file["Y"]
index = 600
single_video = X[index]
cls = LABEL_DECODER[Y[index]]

i = 0
while True:

    img = np.ones((720, 1280))
    img = add_keypoints(img, single_video[:,i % 100])
    img = cv2.putText(img, "frame {}".format(i % 100 + 1), (20, 60), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=2)
    img = cv2.putText(img, cls, (20, 120), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=2)

    cv2.imshow("Image", img)
    cv2.waitKey(100)
    i += 1