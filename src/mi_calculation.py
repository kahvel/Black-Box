import entropy_estimators
import lnc
import pandas as pd
import numpy as np
import itertools


def f(x):
    return sum(x) % 2
    # y = [0,0,1,1,0,1,None,None,None,None,None,None]
    # return int(all(map(lambda a: ((a[0] == a[1]) if a[1] is not None else True), zip(x, y))))


data_generator = itertools.product([0, 1], repeat=12)
all_data = np.array(list(data_generator))
all_values = np.array(list(map(lambda x: [f(x)], all_data)))
print(all_data.shape)
print(all_data)
dummy_data = np.array([[i] for i in range(2**12)])
print(dummy_data.shape)
print(dummy_data)
k = 3
k_cd = 1


def totuple(list_of_lists):
    return tuple(tuple(list) for list in list_of_lists)


def discretize(column):
    bins = np.linspace(-1, 1, 33)
    result = []
    for x in column:
        for i, upper_bound in enumerate(bins[1:]):
            if x < upper_bound and i < len(bins)-1 or i == len(bins)-1 and x == upper_bound:
                result.append(bins[i])
                break
    return tuple(result)


directory = "./test8"
frequency = 500
for j in range(10):
    data = pd.ExcelFile(directory + "/test_" + str(j*frequency) + ".xlsx")
    middd = pd.DataFrame(columns=["x", "y"])
    midd = pd.DataFrame(columns=["x", "y"])
    micd = pd.DataFrame(columns=["x", "y"])
    mi = pd.DataFrame(columns=["x", "y"])
    for i, sheetname in enumerate(data.sheet_names):
        data_frame = np.array(data.parse(sheetname=sheetname))
        print("Sheet: " + str(sheetname))
        print(data_frame.shape)
        # print(lnc.MI.mi_Kraskov(np.column_stack([data_frame, all_data])))
        # print(lnc.MI.mi_LNC(np.column_stack([data_frame, all_data])))
        # mi.loc[i, "x"] = entropy_estimators.mi(data_frame, all_data, k=k)
        # mi.loc[i, "y"] = entropy_estimators.mi(data_frame, all_values, k=k)
        # micd.loc[i, "x"] = entropy_estimators.micd(data_frame.tolist()*(k_cd+1), all_data.tolist()*(k_cd+1), k=k_cd)
        # micd.loc[i, "y"] = entropy_estimators.micd(data_frame.tolist()*(k_cd+1), all_values.tolist()*(k_cd+1), k=k_cd)
        # midd.loc[i, "x"] = entropy_estimators.midd(totuple(data_frame), totuple(all_data))
        # midd.loc[i, "y"] = entropy_estimators.midd(totuple(data_frame), totuple(all_values))
        # discretized_data = tuple(np.histogram(col, bins) for col in data_frame.T)
        discretized_data = totuple(np.transpose(tuple(discretize(col) for col in data_frame.T)))
        # print(data_frame)
        # print(np.array(discretized_data).shape)
        # print((tuple(np.histogram(data_frame.T[0], np.linspace(-1, 1, 33)))))
        # print((tuple(np.histogram(np.array(discretized_data).T[0], np.linspace(-1, 1, 33)))))
        middd.loc[i, "x"] = entropy_estimators.midd(discretized_data, totuple(all_data))
        middd.loc[i, "y"] = entropy_estimators.midd(discretized_data, totuple(all_values))
        # print(str(entropy_estimators.mi(data_frame, dummy_data)))

        # print("Discrete: " + str(entropy_estimators.midd(data_frame, all_data)))
        # print("Mixed: " + str(entropy_estimators.micd(data_frame, all_data)))
    writer = pd.ExcelWriter(directory + "/test_mi_" + str(j*frequency) + ".xlsx", engine="xlsxwriter")
    middd.to_excel(writer, sheet_name="middd")
    # midd.to_excel(writer, sheet_name="midd")
    # micd.to_excel(writer, sheet_name="micd")
    # mi.to_excel(writer, sheet_name="mi")
    writer.close()
