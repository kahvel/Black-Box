import entropy_estimators
import lnc
import pandas as pd
import numpy as np
import itertools


def f(x):
    y = [0,0,1,None,None,None,None,None,None,None,None,None]
    return int(all(map(lambda a: ((a[0] == a[1]) if a[1] is not None else True), zip(x, y))))


data_generator = itertools.product([0, 1], repeat=12)
all_data = np.array(list(data_generator))
all_values = np.array(list(map(lambda x: [f(x)], all_data)))
print(all_data.shape)
print(all_data)
dummy_data = [[i] for i in range(2**12)]

for j in range(1):
    data = pd.ExcelFile("test" + str(j) + ".xlsx")
    for sheetname in data.sheet_names:
        data_frame = np.array(data.parse(sheetname=sheetname))
        print("Sheet: " + str(sheetname))
        print(lnc.MI.mi_Kraskov([data_frame, all_data]), lnc.MI.mi_LNC([data_frame, all_values]))
        print(entropy_estimators.mi(data_frame, all_data), entropy_estimators.mi(data_frame, all_values))
        # print(str(entropy_estimators.micd(data_frame, dummy_data)), str(entropy_estimators.micd(data_frame, all_values)))
        # print(str(entropy_estimators.mi(data_frame, dummy_data)))

        # print("Discrete: " + str(entropy_estimators.midd(data_frame, all_data)))
        # print("Mixed: " + str(entropy_estimators.micd(data_frame, all_data)))
