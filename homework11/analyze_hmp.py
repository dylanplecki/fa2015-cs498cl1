import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class HmpAnalyzeSettings:
    # Analyze Settings
    BlockGroupingSize = 32
    VocabularySize = 10

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
    rawdataframe = None
    inputlist = None

    def __init__(self):
        self.parse_data_files()

    def parse_data_files(self):
        rawdata = []
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
                        (x, y, z) = (int(x), int(y), int(z))
                        zflat.extend([x, y, z])
                        rawdata.append([hmp, gender, vid, timestamp, i, x, y, z])
                        if ((i+1) % HmpAnalyzeSettings.BlockGroupingSize) == 0:
                            segdata.append([hmp, gender, vid, timestamp, segment, zflat])
                            zflat = []
                            segment += 1

        self.rawdataframe = pd.DataFrame(rawdata, columns=self.RawColumnNames)
        self.segdataframe = pd.DataFrame(segdata, columns=self.SegColumnNames)

        # Reshape data vector with copy
        self.inputlist = []
        by_hmp = self.segdataframe.groupby('hmp')
        for hmp_name, group in by_hmp:
            mat = group['z_vec'].values
            zmatrix = np.empty((len(mat), 3*HmpAnalyzeSettings.BlockGroupingSize), dtype=np.float64)
            for i, arr in enumerate(mat):
                for j, val in enumerate(arr):
                    zmatrix[i][j] = val
            self.inputlist.append((hmp_name, zmatrix))


def calculate_k_means(hmp_dataset):
    # Generate k-means clusters
    km = KMeans(n_jobs=1, n_clusters=HmpAnalyzeSettings.VocabularySize)

    cum_zmatrix = np.empty(
        (len(hmp_dataset.segdataframe), 3*HmpAnalyzeSettings.BlockGroupingSize),
        dtype=np.float64
    )

    i = 0
    for (_, zmatrix) in hmp_dataset.inputlist:
        for arr in zmatrix:
            for j, val in enumerate(arr):
                cum_zmatrix[i][j] = val
            i += 1

    km.fit(cum_zmatrix)
    return km


def main():
    if not os.path.exists(HmpAnalyzeSettings.CacheDir):
        os.makedirs(HmpAnalyzeSettings.CacheDir)
    if not os.path.exists(HmpAnalyzeSettings.OutputDir):
        os.makedirs(HmpAnalyzeSettings.OutputDir)

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

    # Predict clusters from input data and plot
    xy_classes = []
    for (hmp_name, zmatrix) in hmp.inputlist:
        k_means_predict = k_means.predict(zmatrix)

        # Plot cluster histogram
        kmpn_fig = plt.figure()
        kmpn_fig_ax = kmpn_fig.add_subplot(111)
        kmpn_fig_ax.hist(k_means_predict, normed=1,
                         range=(0, HmpAnalyzeSettings.VocabularySize), bins=HmpAnalyzeSettings.VocabularySize)
        kmpn_fig_ax.set_title("%s K-Means Histogram" % hmp_name)
        kmpn_fig_ax.set_xlabel("Cluster Index")
        kmpn_fig_ax.set_ylabel("Normalized Frequency")
        kmpn_fig_file = os.path.join(HmpAnalyzeSettings.OutputDir, "%s_k_means_hist.png" % hmp_name)
        kmpn_fig.savefig(kmpn_fig_file, bbox_inches='tight')

        # Generate classification pairs
        X = np.bincount(k_means_predict)
        xy_classes.append((hmp_name, X / len(k_means_predict)))

    #


# Run main from script start
main()
