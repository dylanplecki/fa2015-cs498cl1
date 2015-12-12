import os
import numpy as np
import pandas as pd


class HmpDataset:
    """
        Loads the HMP dataset from disk and provides access to underlying data
    """

    HmpDataDir = "HMP_Dataset"

    def __init__(self, force_reload=False):
        self.TempFile = os.path.join(self.HmpDataDir, 'hmp_dataset.temp.csv')

        if force_reload or (not os.path.isfile(self.TempFile)):
            self.create_temp_file()

        print("Reading temp file...")
        self.data = pd.read_csv(self.TempFile, sep=',')

    def create_temp_file(self):
        if os.path.isfile(self.TempFile):
            os.remove(self.TempFile)

        print("Creating temp file...")
        tempfile = open(self.TempFile, 'w+')
        tempfile.write('hmp,gender,vid,timestamp,tick,x_acc,y_acc,z_acc\n')  # Write headers

        for root, subdirs, files in os.walk(self.HmpDataDir):
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

                # Write contents to temp file
                with open(file, 'r') as f:
                    for i, l in enumerate(f):
                        l = l.rstrip()
                        if not l: continue
                        (x, y, z) = l.split(' ')
                        tempfile.write(','.join([hmp, gender, vid, timestamp, str(i), x, y, z]) + "\n")

###################
# Start Main Body #
###################

dataset = HmpDataset()
