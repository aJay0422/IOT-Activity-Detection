import numpy as np
import pandas as pd


if __name__ == "__main__":
    X_all = pd.read_csv("feature_df_951.csv")
    Y_all = pd.read_csv("label_df_951.csv")
    id_X = X_all.iloc[:,0].to_numpy()
    X_all = X_all.iloc[:,1:].to_numpy()

    id_Y = Y_all.iloc[:,0].to_numpy()
    Y_all = Y_all.iloc[:,1].to_numpy()

    for i in range(len(Y_all)):
        assert id_X[i] == id_Y[i]
    print(X_all.shape)
    print(Y_all.shape)

    np.savez("all_feature_interp951.npz", X=X_all, Y=Y_all, ID=id_X)
    print("saved")