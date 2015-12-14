import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


class HmpAnalyzeSettings:
    # Analyze Settings
    BlockGroupingSize = 32
    VocabularySize = 10
    ClassifierTestSize = 0.25

    # Data Settings
    NoCache = False
    CacheDir = ".cache"
    OutputDir = "output"
    HmpDataDir = "HMP_Dataset"


class HmpDataset:
    """
        Loads the HMP dataset from disk and provides access to underlying data
    """

    # Parameters
    RawColumnNames = ['hmp', 'gender', 'vid', 'timestamp', 'tick', 'x_acc', 'y_acc', 'z_acc']
    SegColumnNames = ['hmp', 'gender', 'vid', 'timestamp', 'segment', 'z_vec']

    # Properties
    segdataframe = None
    inputlist = None

    def __init__(self):
        self.parse_data_files()

    def parse_data_files(self):
        segdata = []

        for root, subdirs, files in os.walk(HmpAnalyzeSettings.HmpDataDir):
            for filename in files:
                file = root + os.sep + filename

                # Parse filename and file contents
                namecomp = filename.split(".")
                if namecomp[1] != 'txt': continue
                components = namecomp[0].split("-")
                timestamp = '-'.join(components[1:7])
                hmp = components[7]
                gender = components[8][0]
                vid = components[8][1:]

                with open(file, 'r') as f:
                    zflat = []
                    segment = 0
                    for i, l in enumerate(f):
                        l = l.rstrip()
                        if not l: continue
                        (x, y, z) = l.split(' ')
                        zflat.extend([int(x), int(y), int(z)])
                        if ((i + 1) % HmpAnalyzeSettings.BlockGroupingSize) == 0:
                            segdata.append([hmp, gender, vid, timestamp, segment, zflat])
                            zflat = []
                            segment += 1

        self.segdataframe = pd.DataFrame(segdata, columns=self.SegColumnNames)

        # Reshape data vector with copy
        self.inputlist = []
        by_hmp = self.segdataframe.groupby('hmp')
        for hmp_name, group in by_hmp:
            by_uid = group.groupby(['gender', 'vid', 'timestamp'])
            samplelist = []
            n = 0
            for uid, group2 in by_uid:
                mat = group2['z_vec'].values
                zmatrix = np.empty((len(mat), 3 * HmpAnalyzeSettings.BlockGroupingSize), dtype=np.float64)
                for i, arr in enumerate(mat):
                    for j, val in enumerate(arr):
                        zmatrix[i][j] = val
                    n += 1
                samplelist.append((uid, zmatrix))
            self.inputlist.append((hmp_name, n, samplelist))


def calculate_k_means(hmp_dataset):
    # Generate k-means clusters
    km = KMeans(n_jobs=1, n_clusters=HmpAnalyzeSettings.VocabularySize)

    cum_zmatrix = np.empty(
        (len(hmp_dataset.segdataframe), 3 * HmpAnalyzeSettings.BlockGroupingSize),
        dtype=np.float64
    )

    i = 0
    for _, _, samplelist in hmp_dataset.inputlist:
        for __, zmatrix in samplelist:
            for arr in zmatrix:
                for j, val in enumerate(arr):
                    cum_zmatrix[i][j] = val
                i += 1

    km.fit(cum_zmatrix)
    return km


def main():
    outputdir = os.path.join(
            HmpAnalyzeSettings.OutputDir,
            "block%d_vocab%d" % (HmpAnalyzeSettings.BlockGroupingSize, HmpAnalyzeSettings.VocabularySize)
    )

    if not os.path.exists(HmpAnalyzeSettings.CacheDir):
        os.makedirs(HmpAnalyzeSettings.CacheDir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    if HmpAnalyzeSettings.NoCache:
        hmp = HmpDataset()
        k_means = calculate_k_means(hmp)
    else:
        hmptempfile = os.path.join(HmpAnalyzeSettings.CacheDir,
                                   "hmp_dataset.b%d.pkl" % HmpAnalyzeSettings.BlockGroupingSize)
        kmeanstempfile = os.path.join(HmpAnalyzeSettings.CacheDir,
                                      "hmp_k_means.v%d.pkl" % HmpAnalyzeSettings.VocabularySize)

        if not os.path.isfile(hmptempfile):
            hmp = HmpDataset()
            fh = open(hmptempfile, "wb")
            fh.truncate()
            pickle.dump(hmp, fh)
            fh.close()
        else:
            fh = open(hmptempfile, "rb")
            hmp = pickle.load(fh)
            fh.close()
        if not os.path.isfile(kmeanstempfile):
            k_means = calculate_k_means(hmp)
            fh = open(kmeanstempfile, "wb")
            fh.truncate()
            pickle.dump(k_means, fh)
            fh.close()
        else:
            fh = open(kmeanstempfile, "rb")
            k_means = pickle.load(fh)
            fh.close()

    x_set = []
    y_set = []

    # Predict clusters from input data and plot
    for hmp_name, n, uniquelist in hmp.inputlist:
        cum_zmatrix = np.empty((n, 3 * HmpAnalyzeSettings.BlockGroupingSize), dtype=np.float64)

        i = 0
        for _, zmatrix in uniquelist:
            for arr in zmatrix:
                for j, val in enumerate(arr):
                    cum_zmatrix[i][j] = val
                i += 1
            kmp = k_means.predict(zmatrix)

            # Generate classification data
            X = np.bincount(kmp, minlength=HmpAnalyzeSettings.VocabularySize)
            x_set.append(X / len(kmp))
            y_set.append(hmp_name)

        kmpn_fig_file = os.path.join(outputdir, "%s_k_means_hist.png" % hmp_name)
        if HmpAnalyzeSettings.NoCache or not os.path.isfile(kmpn_fig_file):

            # Run predication on all HMP data
            k_means_predict = k_means.predict(cum_zmatrix)

            # Plot cluster histogram
            kmpn_fig = plt.figure()
            kmpn_fig_ax = kmpn_fig.add_subplot(111)
            kmpn_fig_ax.hist(k_means_predict, normed=1,
                             range=(0, HmpAnalyzeSettings.VocabularySize), bins=HmpAnalyzeSettings.VocabularySize)
            kmpn_fig_ax.set_title("%s K-Means Histogram" % hmp_name)
            kmpn_fig_ax.set_xlabel("Cluster Index")
            kmpn_fig_ax.set_ylabel("Normalized Frequency")
            kmpn_fig.savefig(kmpn_fig_file, bbox_inches='tight')

    # Generate train/test split for classification
    X_train, X_test, y_train, y_test = train_test_split(x_set, y_set, test_size=HmpAnalyzeSettings.ClassifierTestSize,
                                                        random_state=888)

    # Instantiates an SVM and do some predictions on it
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    score = accuracy_score(y_test, predictions)
    print("Accuracy score of the SVM: ", score)
    cm = confusion_matrix(y_test, predictions)
    print("Confusion matrix: \n", cm)

    # Plot confusion matrix and save to file
    cm_fig = plt.figure()
    cm_fig_ax = cm_fig.add_subplot(111)
    cm_fig_ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cm_fig_ax.set_title("HMP Confusion Matrix")
    cm_fig_ax.set_ylabel('True label')
    cm_fig_ax.set_xlabel('Predicted label')
    cm_fig.colorbar(cm_fig_ax.matshow(cm))
    cm_fig.tight_layout()
    cm_fig.savefig(os.path.join(outputdir, "confusion_matrix.png"), bbox_inches='tight')


# Run main from script start
main()
