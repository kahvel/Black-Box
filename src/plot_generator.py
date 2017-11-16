import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


for j in range(10):
    data = pd.ExcelFile("test2_mi_" + str(j*500) + ".xlsx")
    plt.subplot(5, 2, j + 1)
    plt.xlim((0, 12))
    plt.ylim((0, 1))
    if j < 2 or True:
        plt.title("Epoch " + str(j*500))
    if j % 2 == 0 or True:
        plt.ylabel("I(Y,T)")
    if j > 7:
        plt.xlabel("I(X,T)")
    # plt.axis("equal")
    # plt.gca().set_aspect('equal', adjustable='box')
    for i, sheetname in enumerate(data.sheet_names):
        if sheetname == "middd":
            data_frame = np.array(data.parse(sheetname=sheetname)).T
            # print(data_frame)
            break
    plt.scatter(data_frame[0], data_frame[1], c=np.linspace(0, 1, 8))
plt.show()
