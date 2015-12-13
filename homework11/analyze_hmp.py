import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class HmpAnalyzeSettings:
    # Analyze Settings
    BlockGroupingSize = 32
    VocabularySize = 10

    # Data Settings
    NoCache = False
    CacheDir = ".cache"
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
                        if i > 0 and (i % HmpAnalyzeSettings.BlockGroupingSize == 0):
                            segdata.append([hmp, gender, vid, timestamp, segment, zflat])
                            zflat = []
                            segment += 1

        self.rawdataframe = pd.DataFrame(rawdata, columns=self.RawColumnNames)
        self.segdataframe = pd.DataFrame(segdata, columns=self.SegColumnNames)


###################
# Start Main Body #
###################

hmp = None

if HmpAnalyzeSettings.NoCache:
    hmp = HmpDataset()
else:
    if not os.path.exists(HmpAnalyzeSettings.CacheDir):
        os.makedirs(HmpAnalyzeSettings.CacheDir)

    hmptempfile = os.path.join(HmpAnalyzeSettings.CacheDir, "hmpdataset.pkl")

    if not os.path.isfile(hmptempfile):
        fh = open(hmptempfile, "wb")
        hmp = HmpDataset()
        fh.truncate()
        pickle.dump(hmp, fh)
        fh.close()
    else:
        fh = open(hmptempfile, "rb")
        hmp = pickle.load(fh)
        fh.close()

# Reshape data vector with copy
mat = hmp.segdataframe['z_vec'].values
segdata = np.empty((len(mat), 3*HmpAnalyzeSettings.BlockGroupingSize), dtype=np.float64)
for i, arr in enumerate(mat):
    for j, val in enumerate(arr):
        segdata[i][j] = val

# Generate k-means clusters
km = KMeans(n_jobs=1, n_clusters=HmpAnalyzeSettings.VocabularySize)
km.fit(segdata)
labels = km.labels_
k_means = pd.DataFrame([hmp.segdataframe.index, labels]).T

