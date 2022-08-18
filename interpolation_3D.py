import numpy as np
import os


LABEL_ENCODER = {'no_interaction':0,
                 'open_close_fridge':1,
                 'put_back_item':2,
                 'screen_interaction':3,
                 'take_out_item':4}


def traj_interp(traj, n_frames=100):
    n_samples = traj.shape[0]
    if n_samples == 0:
        raise ValueError("trajectories of length 0")
    result = np.empty((n_frames, 17, 3))
    dest_x = np.linspace(0, 100, n_frames)
    src_x = np.linspace(0, 100, n_samples)
    for i in range(17):
        for j in range(3):
            result[:, i, j] = np.interp(
                dest_x,
                src_x,
                traj[:, i, j]
            )
    return result


def main():
    feature_dir = "feature_archive/3D_features/"
    feature_file_list = os.listdir(feature_dir)
    feature_interp_dir = "feature_archive/3D_features_interp/"
    if not os.path.exists(feature_interp_dir):
        os.mkdir(feature_interp_dir)

    for i, file_name in enumerate(feature_file_list):
        traj = np.load(feature_dir + file_name)
        save_path = feature_interp_dir + file_name
        if os.path.exists(save_path):
            print(i+1, ": jumped")
            continue

        result = traj_interp(traj)
        np.save(save_path, result)
        print(i+1, ": saved")


def check_length(length=100):
    feature_interp_dir = "feature_archive/3D_features_interp/"
    file_list = os.listdir(feature_interp_dir)
    count = 0
    for file in file_list:
        traj = np.load(feature_interp_dir + file)
        length = len(traj)
        if length == length:
            count += 1

    print("{} samples have length {}".format(count, length))
    print("{} samples don't have length {}".format(len(file_list) - count, length))


def get_label():
    feature_interp_dir = "feature_archive/3D_features_interp"
    file_list = os.listdir(feature_interp_dir)
    for i, file_name in enumerate(file_list):
        label = "_".join(file_name.split("_")[:-3])
        file_list[i] = LABEL_ENCODER[label]
    return np.asarray(file_list)


def interp_dataset():
    feature_interp_dir = "feature_archive/3D_features_interp"
    file_list = os.listdir(feature_interp_dir)
    label = get_label()
    X_all = []
    for file_name in file_list:
        traj = np.load(feature_interp_dir + "/" + file_name)   # shape 100, 17, 3
        X_all.append(traj)
    X_all = np.stack(X_all, axis=0)
    np.savez("feature_archive/3D_features_interp951.npz", X=X_all, Y=label)


if __name__ == "__main__":
    feature_dir = "feature_archive/3D_features/"
    file_list = os.listdir(feature_dir)
    traj = np.load(feature_dir + file_list[0])
    print(traj.shape)
    print(traj[0])
    print(np.mean(traj, axis=(1,2)))

