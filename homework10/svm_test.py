import numpy as np
from sklearn.preprocessing import StandardScaler
from svm import SVC
import os
import matplotlib.pyplot as plt

lines = open("wdbc.data").read().split("\n")
lines.pop()
lines = list(map(lambda line: line.split(','), lines))
data = np.array(lines, dtype='object')

# Ignore first column, first column is just IDs
y = data[:, 1]
X = data[:, 2:].astype(float)
X = StandardScaler().fit_transform(X)  # Rescales data

# TODO: Train your SVM based upon different regularization constants

basepath = os.path.dirname(__file__) + "/11_1_output/"
reg_constants = [(1, "1"), (np.exp(-1), "exp_-1"), (np.exp(-2), "exp_-2"), (np.exp(-3), "exp_-3")]

all_fig = plt.figure()
all_fig_ax = all_fig.add_subplot(111)

for reg_const in reg_constants:
    svm = SVC(reg_const[0])
    accuracies = svm.fit(X, y)

    filename = "regc_" + reg_const[1]
    filepath = basepath + filename
    np.savetxt(filepath + ".csv", accuracies, delimiter=",")

    new_fig = plt.figure()
    new_fig_ax = new_fig.add_subplot(111)

    ax_range = range(0, len(accuracies) * 10, 10)
    new_fig_ax.plot(ax_range, accuracies, '-')
    all_fig_ax.plot(ax_range, accuracies, label='RegC ' + reg_const[1])

    new_fig_ax.set_xlabel('Time-Step Number')
    new_fig_ax.set_ylabel('Accuracy (Percentage)')
    new_fig_ax.set_title('Time-Step vs Accuracy Plot for Reg. Constant ' + reg_const[1])
    new_fig.savefig(filepath + '.png', bbox_inches='tight')

all_fig_ax.legend(loc=4)
all_fig_ax.set_xlabel('Time-Step Number')
all_fig_ax.set_ylabel('Accuracy (Percentage)')
all_fig_ax.set_title('Time-Step vs Accuracy Plot for Various Reg. Constants')
all_fig.savefig(basepath + 'timestep_vs_accuracy.png', bbox_inches='tight')
