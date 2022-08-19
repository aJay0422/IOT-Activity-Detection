import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


traj = np.load("feature_archive/3D_features/put_back_item_1_1613086093_1.mp4.npy")
traj[:,:,1] *= -1
# m = np.mean(traj)
# v = np.std(traj)
# traj = (traj - m) / v


def draw_3D(coordinates, legs=False):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.view_init(10, -100)
    ax.set_xlim3d([-0.5, 0.5])
    ax.set_ylim3d([-0.5, 0.5])
    ax.set_zlim3d([-0.5, 0.5])
    coord1 = 0
    coord2 = 2
    coord3 = 1
    # coordinates[:, 1] *= -1
    # coordinates[:, 2] *= 0
    # coordinates[:, 0] *= 0

    ax.plot(coordinates[[0,7,8,9,10],coord1], coordinates[[0,7,8,9,10],coord2], coordinates[[0,7,8,9,10],coord3])   # head to hip
    ax.plot(coordinates[[13, 12, 11, 8],coord1], coordinates[[13, 12, 11, 8],coord2], coordinates[[13, 12, 11, 8],coord3])   # left arm
    ax.plot(coordinates[[16, 15, 14, 8],coord1], coordinates[[16, 15, 14, 8],coord2], coordinates[[16, 15, 14, 8],coord3])   # right arm
    ax.plot(coordinates[[4, 0, 1], coord1], coordinates[[4, 0, 1], coord2], coordinates[[4, 0, 1], coord3])   # hip
    if legs:
        ax.plot(coordinates[[4, 5, 6], coord1], coordinates[[4, 5, 6], coord2], coordinates[[4, 5, 6], coord3])   # left leg
        ax.plot(coordinates[[1, 2, 3], coord1], coordinates[[1, 2, 3], coord2], coordinates[[1, 2, 3], coord3])   # right leg

    return fig


def rotate(coordinates, angle):
    coordinates_new = coordinates.copy()
    center = np.mean(coordinates_new[:,[0,2]], axis=0)
    x_new = coordinates[:,0] * np.cos(angle) - coordinates[:,2] * np.sin(angle)
    z_new = coordinates[:,2] * np.cos(angle) + coordinates[:,0] * np.sin(angle)
    coordinates_new[:,0] = x_new
    coordinates_new[:,2] = z_new
    center_new = np.mean(coordinates_new[:, [0,2]], axis=0)
    coordinates_new[:,[0,2]] -= (center_new - center)

    return coordinates_new


def generate_video(legs=False, if_rotate=True):
    i = 0
    while True:
        coord = traj[i % len(traj)].copy()
        if if_rotate:
            coord = rotate(coord, np.pi / 60 * i)
        fig = draw_3D(coord, legs=legs)
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = image.reshape(int(height), int(width), 3)
        plt.close(fig)
        if if_rotate:
            angle = (int(3 * i) % 360)
            image = cv2.putText(image, "Angle: {}".format(angle), (20, 60), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness=2)
        cv2.imshow("Video", image)
        cv2.waitKey(20)
        i += 1


def augmentation(X_all=None, Y_all=None):
    if X_all is None:
        file = np.load("feature_archive/3D_features_interp951.npz", allow_pickle=True)
        X_all = file["X"]
        Y_all = file["Y"]

    angles = [i * np.pi / 180 for i in [30, -30, 60, -60, 90, -90]]
    X_aug = [X_all]
    Y_aug = [Y_all]
    for angle in angles:
        aug = X_all.copy()
        for i in range(aug.shape[0]):
            for j in range(aug.shape[1]):
                aug[i,j,:,:] = rotate(aug[i,j,:,:], angle=angle)
        X_aug.append(aug)
        Y_aug.append(Y_all)
        print("{} finished".format(angle / np.pi * 180))

    X_aug = np.concatenate(X_aug, axis=0)
    Y_aug = np.concatenate(Y_aug, axis=0)
    print(X_aug.shape)
    print(Y_aug.shape)
    np.savez("feature_archive/3D_features_interp951_aug.npz", X=X_aug, Y=Y_aug)



if __name__ == "__main__":
    # frame = 50
    # legs = False
    # initial = traj[frame].copy()
    # draw_3D(initial, legs=legs)
    # for i in [6, 3, 2, 1.5, 1.2, 1]:
    #     coord = rotate(initial, np.pi / i)
    #     draw_3D(coord, legs=legs)


    generate_video(legs=True, if_rotate=True)

    # augmentation()