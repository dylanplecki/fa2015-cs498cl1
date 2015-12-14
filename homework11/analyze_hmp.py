import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


class HmpAnalyzeSettings:
    # Single-Analyze Settings
    BlockGroupingSize = 64
    VocabularySize = 50
    ClassifierTestSize = 0.25

    # Multi-Analyze Settings
    MultiAnalyze = True
    TestBlockSizes = [16, 32, 64]
    TestVocabularySizes = [10, 20, 30, 40, 50]

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
    else:
        hmptempfile = os.path.join(HmpAnalyzeSettings.CacheDir,
                                   "hmp_dataset.b%d.pkl" % HmpAnalyzeSettings.BlockGroupingSize)
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

    # Calculate k-means for dataset
    k_means = calculate_k_means(hmp)

    # Collection lists for classification
    x_set = []
    y_set = []

    # Initialize signal plots
    signal_fig = plt.figure()
    signal_x_fig_ax = signal_fig.add_subplot(311)
    signal_y_fig_ax = signal_fig.add_subplot(312)
    signal_z_fig_ax = signal_fig.add_subplot(313)
    signal_x_fig_ax.set_xlabel("Time")
    signal_y_fig_ax.set_xlabel("Time")
    signal_z_fig_ax.set_xlabel("Time")
    signal_x_fig_ax.set_ylabel("$X$ Acceleration")
    signal_y_fig_ax.set_ylabel("$Y$ Acceleration")
    signal_z_fig_ax.set_ylabel("$Z$ Acceleration")
    signal_x_fig_ax.set_title("$X$ Acceleration Cluster Signals for BlockSize=%d, Clusters=%d" %
                              (HmpAnalyzeSettings.BlockGroupingSize, HmpAnalyzeSettings.VocabularySize))
    signal_y_fig_ax.set_title("$Y$ Acceleration Cluster Signals for BlockSize=%d, Clusters=%d" %
                              (HmpAnalyzeSettings.BlockGroupingSize, HmpAnalyzeSettings.VocabularySize))
    signal_z_fig_ax.set_title("$Z$ Acceleration Cluster Signals for BlockSize=%d, Clusters=%d" %
                              (HmpAnalyzeSettings.BlockGroupingSize, HmpAnalyzeSettings.VocabularySize))

    # Plot cluster centers as signals
    for cluster in k_means.cluster_centers_:
        signal_x_fig_ax.plot(cluster[0::3])
        signal_y_fig_ax.plot(cluster[1::3])
        signal_z_fig_ax.plot(cluster[2::3])

    # Save signal plot
    signal_fig.tight_layout()
    signal_fig.savefig(os.path.join(outputdir, "signal_plot.png"), bbox_inches='tight')

    # Predict clusters from input data and plot
    p = 0
    kmpn_fig = plt.figure()
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

        # Run predication on all HMP data
        k_means_predict = k_means.predict(cum_zmatrix)

        # Plot cluster histogram
        kmpn_fig_ax = kmpn_fig.add_subplot(4, 4, p + 1)
        kmpn_fig_ax.hist(k_means_predict, normed=1,
                         range=(0, HmpAnalyzeSettings.VocabularySize), bins=HmpAnalyzeSettings.VocabularySize)
        kmpn_fig_ax.set_title("%s K-Means Histogram" % hmp_name)
        kmpn_fig_ax.set_xlabel("Cluster Index")
        kmpn_fig_ax.set_ylabel("Normalized Frequency")

        p += 1

    # Save k-means histograms
    kmpn_fig.suptitle("K-Means Histograms for BlockSize=%d, Clusters=%d" %
                      (HmpAnalyzeSettings.BlockGroupingSize, HmpAnalyzeSettings.VocabularySize))
    kmpn_fig.set_size_inches(24, 24)
    kmpn_fig.savefig(os.path.join(outputdir, "k_means_histograms.png"), bbox_inches='tight')

    # Generate train/test split for classification
    X_train, X_test, y_train, y_test = train_test_split(x_set, y_set, test_size=HmpAnalyzeSettings.ClassifierTestSize,
                                                        random_state=randint(90, 1000))

    # Instantiates an SVM and do some predictions on it
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Generate results
    score = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    # Save results
    file = open(os.path.join(outputdir, "results.txt"), "w")
    file.write("SVM Results for BlockSize=%d, Clusters=%d\n\n" %
               (HmpAnalyzeSettings.BlockGroupingSize, HmpAnalyzeSettings.VocabularySize))
    file.write("Accuracy score of the SVM: %f\n" % score)
    file.write("Confusion matrix (text-form):\n%s\n" % str(cm))
    file.close()

    # Plot confusion matrix and save to file
    cm_fig = plt.figure()
    cm_fig_ax = cm_fig.add_subplot(111)
    cm_fig_ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cm_fig_ax.set_title("HMP Confusion Matrix for BlockSize=%d, Clusters=%d" %
                        (HmpAnalyzeSettings.BlockGroupingSize, HmpAnalyzeSettings.VocabularySize))
    cm_fig_ax.set_ylabel('True Label')
    cm_fig_ax.set_xlabel('Predicted Label')
    cm_fig.colorbar(cm_fig_ax.matshow(cm))
    cm_fig.tight_layout()
    cm_fig.savefig(os.path.join(outputdir, "confusion_matrix.png"), bbox_inches='tight')


# Run main from script start
if HmpAnalyzeSettings.MultiAnalyze:
    for blocksize in HmpAnalyzeSettings.TestBlockSizes:
        for clustersize in HmpAnalyzeSettings.TestVocabularySizes:
            HmpAnalyzeSettings.BlockGroupingSize = blocksize
            HmpAnalyzeSettings.VocabularySize = clustersize
            main()
else:
    main()
