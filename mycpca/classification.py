import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from mycpca.cpca import CPCA


FEATURE_ARCHIVE = "../feature_archive/"
LABEL_ENCODER = {'no_interaction': 0,
                 'open_close_fridge': 1,
                 'put_back_item': 2,
                 'screen_interaction': 3,
                 'take_out_item': 4}


def main1(test_ratio=0.15, seed=20220703):
    """CPCA on interpolated data(same sequence length of sample"""
    summary = {}
    # prepare train and test data
    all_data = np.load(FEATURE_ARCHIVE + "all_feature_interp971.npz", allow_pickle=True)
    X_all = all_data["X"]
    Y_all = all_data["Y"]

    # set random seed
    np.random.seed(seed)

    # split data
    n_samples = X_all.shape[0]
    test_size = int(n_samples * test_ratio)
    perm = np.random.permutation(n_samples)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]
    X_train = X_all[train_idx]
    Y_train = Y_all[train_idx]
    X_test = X_all[test_idx]
    Y_test = Y_all[test_idx]

    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # fit a baseline model
    parameters = {"C": [0.001, 0.003, 0.01, 0.03, 0.01, 0.03, 1, 3]}
    clf = GridSearchCV(estimator=LogisticRegression(max_iter=1000, solver="newton-cg"),
                       param_grid=parameters)
    clf.fit(X_train_scaled, Y_train)
    print(clf.cv_results_)
    print(clf.best_params_)
    best_C = clf.best_params_["C"]
    lr_base = LogisticRegression(max_iter=1000,
                                 solver="newton-cg",
                                 C=best_C)
    lr_base.fit(X_train_scaled, Y_train)
    Y_train_pred = lr_base.predict(X_train_scaled)
    Y_test_pred = lr_base.predict(X_test_scaled)
    train_acc_base = np.mean(Y_train == Y_train_pred)
    test_acc_base = np.mean(Y_test == Y_test_pred)
    train_f1_base = f1_score(Y_train, Y_train_pred, average="macro")
    test_f1_base = f1_score(Y_test, Y_test_pred, average="macro")
    summary["train_acc_base"] = train_acc_base
    summary["test_acc_base"] = test_acc_base
    summary["train_f1_base"] = train_f1_base
    summary["test_f1_base"] = test_f1_base

    # reshape data to (n_samples, n_features, T_k)
    X_train_scaled = X_train_scaled.reshape(-1, 100, 34).transpose(0, 2, 1)
    X_test_scaled = X_test_scaled.reshape(-1, 100, 34).transpose(0, 2, 1)

    #CPCA
    cpca_directions = [i for i in range(5, 35, 5)]
    train_accs_cpca = []
    train_f1s_cpca = []
    test_accs_cpca = []
    test_f1s_cpca = []
    for n_directions in cpca_directions:
        cpca = CPCA(n_directions=n_directions)
        X_train_scaled_cpca = cpca.fit_transform(X_train_scaled)
        X_test_scaled_cpca = cpca.transform(X_test_scaled)
        # print("{} features are reduced to {} features using CPCA".format(X_train.shape[1], X_train_cpca.shape[1]))
        # print("-" * 100)

        # reshape data
        X_train_scaled_cpca = X_train_scaled_cpca.reshape(X_train.shape[0], -1)
        X_test_scaled_cpca = X_test_scaled_cpca.reshape(X_test.shape[0], -1)

        # fit a CPCA Logistic Model
        clf = GridSearchCV(estimator=LogisticRegression(max_iter=1000, solver="newton-cg"),
                           param_grid=parameters)
        clf.fit(X_train_scaled_cpca, Y_train)
        print(clf.cv_results_)
        print(clf.best_params_)
        best_C = clf.best_params_["C"]
        lr_cpca = LogisticRegression(max_iter=1000, solver="newton-cg", C=best_C)
        lr_cpca.fit(X_train_scaled_cpca, Y_train)
        Y_train_pred = lr_cpca.predict(X_train_scaled_cpca)
        Y_test_pred = lr_cpca.predict(X_test_scaled_cpca)
        train_acc_cpca = np.mean(Y_train == Y_train_pred)
        test_acc_cpca = np.mean(Y_test == Y_test_pred)
        train_f1_cpca = f1_score(Y_train, Y_train_pred, average="macro")
        test_f1_cpca = f1_score(Y_test, Y_test_pred, average="macro")
        train_accs_cpca.append(train_acc_cpca)
        test_accs_cpca.append(test_acc_cpca)
        train_f1s_cpca.append(train_f1_cpca)
        test_f1s_cpca.append(test_f1_cpca)

    summary["train_accs_cpca"] = train_accs_cpca
    summary["test_accs_cpca"] = test_accs_cpca
    summary["train_f1s_cpca"] = train_f1s_cpca
    summary["test_f1s_cpca"] = test_f1s_cpca

    print("Finished")
    return summary




if __name__ == "__main__":
    main1()
    df = pd.DataFrame(columns=["cpca_directions",
                               "Train Acc", "Train F1",
                               "Test Acc", "Test F1"])
    df.loc[0, "cpca_directions"] = "baseline"
    cpca_directions = [i for i in range(5, 35, 5)]
    for i, cpca_direction in enumerate(cpca_directions):
        df.loc[i + 1, "cpca_directions"] = cpca_direction

    summarys = []
    n = 20   # repeat n times
    for i in range(n):
        summary = main1(seed=20220703 + i)
        summarys.append(summary)

    # baseline acc
    tmp_train = []
    tmp_test = []
    for i in range(n):
        tmp_train.append(summarys[i]["train_acc_base"])
        tmp_test.append(summarys[i]["test_acc_base"])
    m_train = np.mean(tmp_train)
    m_test = np.mean(tmp_test)
    std_train = np.std(tmp_train)
    std_test = np.std(tmp_test)
    df.loc[0, "Train Acc"] = "{:.2f}(±{:.2f})%".format(m_train * 100, std_train * 2 * 100 / np.sqrt(n))
    df.loc[0, "Test Acc"] = "{:.2f}(±{:.2f})%".format(m_test * 100, std_test * 2 * 100 / np.sqrt(n))

    # baseline f1
    mp_train = []
    tmp_test = []
    for i in range(n):
        tmp_train.append(summarys[i]["train_f1_base"])
        tmp_test.append(summarys[i]["test_f1_base"])
    m_train = np.mean(tmp_train)
    m_test = np.mean(tmp_test)
    std_train = np.std(tmp_train)
    std_test = np.std(tmp_test)
    df.loc[0, "Train F1"] = "{:.2f}(±{:.2f})%".format(m_train * 100, std_train * 2 * 100 / np.sqrt(n))
    df.loc[0, "Test F1"] = "{:.2f}(±{:.2f})%".format(m_test * 100, std_test * 2 * 100 / np.sqrt(n))

    # CPCA acc
    tmp_train = []
    tmp_test = []
    for i in range(n):
        tmp_train.append(summarys[i]["train_accs_cpca"])
        tmp_test.append(summarys[i]["test_accs_cpca"])
    tmp_train = np.stack(tmp_train)
    tmp_test = np.stack(tmp_test)
    m_train = np.mean(tmp_train, axis=0)
    m_test = np.mean(tmp_test, axis=0)
    std_train = np.std(tmp_train, axis=0)
    std_test = np.std(tmp_test, axis=0)
    for i, _ in enumerate(cpca_directions):
        df.loc[i + 1, "Train Acc"] = "{:.2f}(±{:.2f})%".format(m_train[i] * 100, std_train[i] * 2 * 100 / np.sqrt(n))
        df.loc[i + 1, "Test Acc"] = "{:.2f}(±{:.2f})%".format(m_test[i] * 100, std_test[i] * 2 * 100 / np.sqrt(n))

    # CPCA F1
    tmp_train = []
    tmp_test = []
    for i in range(n):
        tmp_train.append(summarys[i]["train_f1s_cpca"])
        tmp_test.append(summarys[i]["test_f1s_cpca"])
    tmp_train = np.stack(tmp_train)
    tmp_test = np.stack(tmp_test)
    m_train = np.mean(tmp_train, axis=0)
    m_test = np.mean(tmp_test, axis=0)
    std_train = np.std(tmp_train, axis=0)
    std_test = np.std(tmp_test, axis=0)
    for i, _ in enumerate(cpca_directions):
        df.loc[i + 1, "Train F1"] = "{:.2f}(±{:.2f})%".format(m_train[i] * 100, std_train[i] * 2 * 100 / np.sqrt(n))
        df.loc[i + 1, "Test F1"] = "{:.2f}(±{:.2f})%".format(m_test[i] * 100, std_test[i] * 2 * 100 / np.sqrt(n))

    print(df)
    df.to_csv("summary_table_CPCA.csv")