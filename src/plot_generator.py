import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directory = "./test1_12"
frequency = 500
mi_estimation_method = "middd"
bins = "1_33"
# bins = "pi/2_333"

if bins == "1_33":
    file_name = "/test_mi_"
elif bins == "pi/2_333":
    file_name = "/test_mi_new_bins_"
else:
    raise Exception

for j in range(10):
    data = pd.ExcelFile(directory + file_name + str(j*frequency) + ".xlsx")
    plt.subplot(5, 2, j + 1)
    plt.xlim((0, 12))
    plt.ylim((-0.1, 1.1))
    if j < 2 or True:
        plt.title("Epoch " + str(j*frequency))
    if j % 2 == 0 or True:
        plt.ylabel("I(Y,T)")
    if j > 7:
        plt.xlabel("I(X,T)")
    # plt.axis("equal")
    # plt.gca().set_aspect('equal', adjustable='box')
    for i, sheetname in enumerate(data.sheet_names):
        if sheetname == mi_estimation_method:
            data_frame = np.array(data.parse(sheetname=sheetname)).T
            # print(data_frame)
            break
    plt.scatter(data_frame[0], data_frame[1], c=np.linspace(0, 1, 8))
plt.show()


# weight decay (L2 regularisation)
# Two functions with some shared and some unique information
# how weights change?
# Partial information theory
# Noisy copy of label
# Just with more epochs (1)
# moitor gradients and weights (2)
# normalised gradients
# Two functions (4) or one function and calculate partial stuff yourself
# Noisy copy (5)

# information decomposition (4)
# Divide 12 inputs to 2 xors, that always have the same bit (fewer  inputs).

# analog to digital, relu.

# check weights when learning XOR. Prove something.
