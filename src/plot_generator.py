import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directory = "./test3_1"
frequency = 500
mi_estimation_method = "middd"
# bins = "1_33"
bins = "pi/2_333"

if bins == "1_33":
    file_name = "/test_mi_"
elif bins == "pi/2_333":
    file_name = "/test_mi_new_bins_"
else:
    raise Exception


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, verts=verts, hold=hold, **kwargs)


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
            print(i, sheetname)
            print(data_frame.T)
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





# Check what if happens if you sample 4096 samples randomly etc.
# Its not the number of samples but probably probability distribution.

# Check if 0.5 makes sense.
# p - true label
# q - output

# L(x, N) = p(x) log (1/q(x)) + (1-p(x)) log (1/(1-q(x)))
# N - network?

# L'(x,N) = |round(g(x))-p(x)|
#


# Activation larger range, if all the samples.


# what is the minimum number of samples you need to approximate the function.
# Or how to sample to get better results.

# the amount of outliers is large.

# change the function to sum of variables.

# See what happes without outliers and with outliers (from one side or the other).

# possible problems: more outliers,


# stable distributions.

# increasing the number of inputs.


# information distribution in the input - how it affects the compression. How it affects the ability to learn.
# Dimensionality is also interesting.


# take a function that does not have synergistic information and see if it does the same.
# Outliers more general. Then it should not matter how much synergistic etc information there is.

# sample 3000 samples, but differently (for example, taking a random number between 0 and 12 and this will be the number of ones)
# or from 0 to 24 (20, 17, etc) and mod 12.
# or from all the vectors that have certain number of ones take them a certain number.


# Concentration of measure - sampling randomly from a function

# Dirk - if you go up with n, it will become even more difficult to learn XOR.
# How things scale. Outliers become exponentially more rare.

# Checkerboard pattern. Tradeoff between larger range and pattern squeezed together.

# if small number of variance has a lot of influence, then the fourier transform is concentrated on small sets.
# if large number, then more flat.
# Sample a function that has a certain influence?

# majority function - everybody has least influence.

# 5 boxes of input which we XOR, each box has majority rule inside it. Parameter of influence - ratio between XOR and majority.
# manifold - fold. ones are points in a space. Each layer applies a transformation.

# Only non-negative weights?

# Parity with one (maybe two) hidden layers, with (n log n) nodes.

# training a network gives you a random local minimum.

# Create objective, like linear programming task.

# How things scale - control everything change one thing.

# If information is spread, it is harder for the network to learn (maybe needs more hidden layers). Concentrated - good. Folding is easier.


# Add like 2 nodes to each layer, Dirks conjecture is that then it would work with all 4096 inputs.
# how to we then test the function on unseen data?

# 3000 for 12. What is the 3000 for larger values?
# Find the number of inputs for which the "normal distribution" width is 13.

# Using larger neural network to find a
# Trying larger XORs to see how the learnibility scales? Outscale the input so that the
# Making it larger to be able to put in all the 4096 samples and it would learn it.


# batch normalisation - multiplying (weight?) with a factor

#first try with







# Compare gradients
# Batch size vs
