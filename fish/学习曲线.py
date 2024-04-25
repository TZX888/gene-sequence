import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1,1.0,5)):
    train_size, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

digits = load_digits()

title = "Learning Curves (GaussianNB)"
X= np.load(file="/home/u2208283040/tzx/cscwd/对比/data_X_11.npy")
y_fish=np.load(file="/home/u2208283040/tzx/cscwd/对比/data_y_11.npy")

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
plot_learning_curve(model, title, X, y_fish, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)
plt.savefig('/home/u2208283040/tzx/cscwd/k_11/学习曲线/Random_Forest.eps')
plt.savefig('/home/u2208283040/tzx/cscwd/k_11/学习曲线/Random_Forest.png')
plt.show()