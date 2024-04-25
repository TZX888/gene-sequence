import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

# #############################################################################
for i in range(0,1):
    X_train=np.load(file="/home/u2208283040/tzx/cscwd/k_11/AUC/X_"+str(i)+".npy")
    X_test=np.load(file="/home/u2208283040/tzx/cscwd/k_11/AUC/Y_"+str(i)+".npy")
    y_train=np.load(file="/home/u2208283040/tzx/cscwd/k_11/data/y_train.npy")


    # np.random.seed(12)
    # np.random.shuffle(X_train)
    np.random.seed(12)
    np.random.shuffle(y_train)


    from sklearn.naive_bayes import MultinomialNB
    # from powershap import PowerShap
    # from sklearn.svm import SVC
    import shap
    import joblib
    #============================Random_Forest=========================
    from sklearn.ensemble import RandomForestClassifier


    cv = StratifiedKFold(n_splits=10)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        classifier.fit(X_train, y_train)
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X_train[test],
            y_train[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print(mean_auc)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")

    plt.show()
