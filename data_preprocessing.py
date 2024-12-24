import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

folders = ["AR drone", "Background ", "AIR_ON", "DIS_FY", "DIS_ON", "INS_FY", "INS_HO", "INS_ON",
           "MIN_FY", "MIN_HO", "MIN_ON", "MP1_FY", "MP1_HO", "MP1_ON", "MP2_FY", "MP2_HO", "MP2_ON", "PHA_FY",
           "PHA_HO", "PHA_ON"]

# path_to_export = "F:/Drone_csv/Clean/"
path_to_export1 = "/home/ubuntu/Downloads/DroneDetect_V2/CLEAN/"

for i in folders:
    for (root, dirs, file) in os.walk('/home/ubuntu/Downloads/DroneDetect_V2/CLEAN/' + i):
        # print(file)
        # new_p = path_to_export+i
        new_p1 = path_to_export1 + i
        # os.mkdir(new_p)
        # os.mkdir(new_p1)
        for f in file:
            path = ('/home/ubuntu/Downloads/DroneDetect_V2/CLEAN/' + i + '/' + f)
            print(path)
            # print(type(path))
            f1 = open(path, "rb")  # open file
            data1 = np.fromfile(f1, dtype="float32", count=240000000)  # read the data into numpy array
            f1.close()
            data1 = data1.astype(np.float32).view(np.complex64)  # view as complex
            data = data1.view(np.float32)  # convert into two columns of real numbers
            del data1
            data_norm = (data - np.mean(data)) / (np.sqrt(np.var(data)))  # normalise
            del data
            newarr = np.array_split(data_norm, 400)
            # split the array, 100 will equate to a sample length of 20ms
            # 400 will equate to a sample length of 5ms
            del data_norm
            df3 = pd.DataFrame(newarr)
            del newarr
            try:
                df3 = df3.drop(['Unnamed: 0'], axis=1)
            except:
                print(" / ")

            df3.to_hdf(path_to_export1 + i + '/' + f[:11] + '.h5', 'data')
            del df3
        print()

