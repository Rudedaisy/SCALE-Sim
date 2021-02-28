# Analyze data from the RECTANGLES.sh experiment
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os.path
import math

"""
paths = ["outputs/RECTANGLES_ResNet50_ImageNet_1024_4_baseline/",
         "outputs/RECTANGLES_ResNet50_ImageNet_512_8_baseline/",
         "outputs/RECTANGLES_ResNet50_ImageNet_256_16_baseline/",
         "outputs/RECTANGLES_ResNet50_ImageNet_128_32_baseline/",
         "outputs/RECTANGLES_ResNet50_ImageNet_64_64_baseline/",
         "outputs/RECTANGLES_ResNet50_ImageNet_32_128_baseline/",
         "outputs/RECTANGLES_ResNet50_ImageNet_16_256_baseline/",
         "outputs/RECTANGLES_ResNet50_ImageNet_8_512_baseline/",
         "outputs/RECTANGLES_ResNet50_ImageNet_4_1024_baseline/"]
network = "ResNet50_ImageNet_baseline"
"""
paths = ["outputs/RECTANGLES_ResNet50_ImageNet_1024_4_PENNIv3/",
         "outputs/RECTANGLES_ResNet50_ImageNet_512_8_PENNIv3/",
         "outputs/RECTANGLES_ResNet50_ImageNet_256_16_PENNIv3/",
         "outputs/RECTANGLES_ResNet50_ImageNet_128_32_PENNIv3/",
         "outputs/RECTANGLES_ResNet50_ImageNet_64_64_PENNIv3/",
         "outputs/RECTANGLES_ResNet50_ImageNet_32_128_PENNIv3/",
         "outputs/RECTANGLES_ResNet50_ImageNet_16_256_PENNIv3/",
         "outputs/RECTANGLES_ResNet50_ImageNet_8_512_PENNIv3/",
         "outputs/RECTANGLES_ResNet50_ImageNet_4_1024_PENNIv3/"]
network = "ResNet50"
#"""


utils = []
cycles = []
for exp_idx in range(len(paths)):
    f = open(paths[exp_idx] + network + "_cycles.csv", 'r')

    util = []
    cycle = []
    for idx, layer in enumerate(f):
        if idx == 0:
            continue
        elems = layer.strip().split(',')
        cycle.append(int(elems[1]))
        util.append(float(elems[2]))
    utils.append(util)
    cycles.append(cycle)

    f.close()

utils = np.array(utils)
cycles = np.array(cycles)
print(np.shape(utils))

sram_accesses = []
for exp_idx in range(len(paths)):
    #f = open(paths[exp_idx] + network + "_avg_bw.csv", 'r')
    f = open(paths[exp_idx] + network + "_sram_accesses.csv", 'r')

    sram_access = []
    for idx, layer in enumerate(f):
        if idx == 0:
            continue
        elems = layer.strip().split(',')
        #sum_bw = float(elems[7]) + float(elems[8])
        #sram_access.append(sum_bw * cycles[exp_idx, idx-1])
        sram_access.append(int(elems[1]) + int(elems[2]) + int(elems[4]))
    sram_accesses.append(sram_access)

    f.close()

sram_accesses = np.array(sram_accesses)

x = list(range(np.shape(utils)[1]))
# --- utils ---
ax = plt.subplot()
ax.plot(x, utils[4,:], marker='o')
ax.set_title("Utilization")
plt.grid(axis='x')
plt.show()

# --- layer_impacts ---
layer_impacts_sram = [float(i)/sum(sram_accesses[4,:]) for i in sram_accesses[4,:]]
layer_impacts_cycles = [float(i)/sum(cycles[4,:]) for i in cycles[4,:]]
ax = plt.subplot()
ax.plot(x, layer_impacts_sram, marker='o')
ax.plot(x, layer_impacts_cycles, marker='o')
ax.set_title("Impact of layer compared to total")
ax.legend(["SRAM Accesses", "Cycles"])
plt.grid(axis='x')
plt.show()

# --- util_maxes_idx ---
util_maxes_idx = np.argmax(utils, axis=0)

# --- sram_mins_idx ---
sram_mins_idx = np.argmin(sram_accesses, axis=0)

# --- cycle_mins_idx ---
cycle_mins_idx = np.argmin(cycles, axis=0)

ax = plt.subplot()
#ax.plot(x, util_maxes_idx, marker='o')
ax.plot(x, sram_mins_idx, marker='o')
ax.plot(x, cycle_mins_idx, marker='o')
ax.set_title("Best Shapes")
ax.legend(["sram_mins_idx", "cycle_mins_idx"])
plt.grid(axis='x')
plt.show()

# --- util_diffs ---
util_diffs = []
for layer in range(np.shape(utils)[1]):
    #util_diffs.append(utils[util_maxes_idx[layer], layer] - utils[4, layer])
    util_diffs.append(utils[util_maxes_idx[layer], layer])
util_diffs = [float(i)/util_diffs[4] for i in util_diffs]

#ax = plt.subplot()
#ax.plot(x, util_diffs, marker='o')
#ax.set_title("util_diffs")
#plt.grid(axis='x')
#plt.show()

# --- sram_diffs ---
sram_diffs = []
for layer in range(np.shape(utils)[1]):
    #sram_diffs.append(sram_accesses[4, layer] - sram_accesses[sram_mins_idx[layer], layer])
    sram_diffs.append(sram_accesses[sram_mins_idx[layer], layer])
sram_diffs = [sram_accesses[4,idx]/float(i) for idx,i in enumerate(sram_diffs)]

#ax = plt.subplot()
#ax.plot(x, sram_diffs, marker='o')
#ax.set_title("sram_diffs")
#plt.grid(axis='x')
#plt.show()

# --- cycle_diffs ---
cycle_diffs = []
for layer in range(np.shape(utils)[1]):
    #cycle_diffs.append(cycles[4, layer] - cycles[cycle_mins_idx[layer], layer])
    cycle_diffs.append(cycles[cycle_mins_idx[layer], layer])
cycle_diffs = [cycles[4,idx]/float(i) for idx, i in enumerate(cycle_diffs)]

ax = plt.subplot()
#ax.plot(x, util_diffs, marker='o')
ax.plot(x, sram_diffs, marker='o')
ax.plot(x, cycle_diffs, marker='o')
ax.set_title("Diffs")
ax.legend(["sram_diffs", "cycle_diffs"])
plt.yscale("log")
plt.grid(axis='x')
plt.show()

# --- throughput and sram cost of square matrix vs matrix with max utilization (at each layer) ---
print("Nominal latency of square array: {}".format(sum(cycles[4])))
smartsum = 0
for layer in range(np.shape(utils)[1]):
    smartsum += cycles[util_maxes_idx[layer], layer]
print("Latency of dynamic array: {}".format(smartsum))
print("Speedup: {}".format(float(sum(cycles[4])) / float(smartsum)))

print("---")

print("Nominal number of SRAM accesses of square array: {}".format(sum(sram_accesses[4])))
smartsum = 0
for layer in range(np.shape(utils)[1]):
    smartsum += sram_accesses[sram_mins_idx[layer], layer]
print("Number of SRAM accesses of dynamic array: {}".format(smartsum))
print("Efficiency: {}".format(float(sum(sram_accesses[4])) / float(smartsum)))


# --- overall ---
displacements = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4]
legend = ["1024x4", "512x8", "256x16", "128x32", "64x64", "32x128", "16x256", "8x512", "4x1024"]
ax = plt.subplot()
for exp_idx in range(len(paths)):
    #ax.bar(x+([displacements[exp_idx]]*np.shape(utils)[1]), utils[exp_idx], width=0.1, align='center')
    ax.plot(x, utils[exp_idx])
#ax.autoscale(tight=True)
ax.legend(legend)
plt.show()
