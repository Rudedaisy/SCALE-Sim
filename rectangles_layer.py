# Analyze data from the RECTANGLES.sh experiment
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os.path
import math

plt.rcParams.update({'font.size': 12})

paths = []
networks = []

path = ["outputs/RECTANGLES_indiv_conv3_5x80/",
         "outputs/RECTANGLES_indiv_conv3_10x40/",
         "outputs/RECTANGLES_indiv_conv3_20x20/",
         "outputs/RECTANGLES_indiv_conv3_40x10/",
         "outputs/RECTANGLES_indiv_conv3_80x5/"]
network = "conv3"
paths.append(path)
networks.append(network)

path = ["outputs/RECTANGLES_indiv_conv3_deep_5x80/",
         "outputs/RECTANGLES_indiv_conv3_deep_10x40/",
         "outputs/RECTANGLES_indiv_conv3_deep_20x20/",
         "outputs/RECTANGLES_indiv_conv3_deep_40x10/",
         "outputs/RECTANGLES_indiv_conv3_deep_80x5/"]
network = "conv3_deep"
paths.append(path)
networks.append(network)

path = ["outputs/RECTANGLES_indiv_bottleneck_5x80/",
         "outputs/RECTANGLES_indiv_bottleneck_10x40/",
         "outputs/RECTANGLES_indiv_bottleneck_20x20/",
         "outputs/RECTANGLES_indiv_bottleneck_40x10/",
         "outputs/RECTANGLES_indiv_bottleneck_80x5/"]
network = "bottleneck"
paths.append(path)
networks.append(network)

path = ["outputs/RECTANGLES_indiv_pointwise_5x80/",
         "outputs/RECTANGLES_indiv_pointwise_10x40/",
         "outputs/RECTANGLES_indiv_pointwise_20x20/",
         "outputs/RECTANGLES_indiv_pointwise_40x10/",
         "outputs/RECTANGLES_indiv_pointwise_80x5/"]
network = "pointwise"
paths.append(path)
networks.append(network)

path = ["outputs/RECTANGLES_indiv_resid_5x80/",
         "outputs/RECTANGLES_indiv_resid_10x40/",
         "outputs/RECTANGLES_indiv_resid_20x20/",
         "outputs/RECTANGLES_indiv_resid_40x10/",
         "outputs/RECTANGLES_indiv_resid_80x5/"]
network = "resid"
paths.append(path)
networks.append(network)

#"""
path = ["outputs/RECTANGLES_indiv_dsc_5x80/",
         "outputs/RECTANGLES_indiv_dsc_10x40/",
         "outputs/RECTANGLES_indiv_dsc_20x20/",
         "outputs/RECTANGLES_indiv_dsc_40x10/",
         "outputs/RECTANGLES_indiv_dsc_80x5/"]
network = "dsc"
paths.append(path)
networks.append(network)
#"""

layerNames = ["conv3", "conv3_deep", "bottleneck", "pointwise", "resid", "dsc"]
widths = ["5x80", "10x40", "20x20", "40x10", "80x5"]

def grabCycleUtil(path, network):
    utils = []
    cycles = []
    for exp_idx in range(len(widths)):
        f = open(path[exp_idx] + network + "_cycles.csv", 'r')
        
        util = 0
        cycle = 0
        for idx, layer in enumerate(f):
            if idx == 0:
                continue
            elems = layer.strip().split(',')
            cycle = int(elems[1])
            util = float(elems[2])
        utils.append(util)
        cycles.append(cycle)
        
        f.close()

    #print(utils)
    utils = [float(i)/max(utils) for i in utils]
    cycles = [float(i)/max(cycles) for i in cycles]
        
    utils = np.array(utils)
    cycles = np.array(cycles)
    #print(np.shape(utils))
    return cycles, utils

def grabSRAM(path, network):
    sram_accesses = []
    for exp_idx in range(len(widths)):
        #f = open(paths[exp_idx] + network + "_avg_bw.csv", 'r')
        f = open(path[exp_idx] + network + "_sram_accesses.csv", 'r')
        
        sram_access = 0
        for idx, layer in enumerate(f):
            if idx == 0:
                continue
            elems = layer.strip().split(',')
            #sum_bw = float(elems[7]) + float(elems[8])
            #sram_access.append(sum_bw * cycles[exp_idx, idx-1])
            sram_access = int(elems[0]) + int(elems[2]) + int(elems[3])
        sram_accesses.append(sram_access)
        
        f.close()

    sram_accesses = [float(i)/max(sram_accesses) for i in sram_accesses]
        
    sram_accesses = np.array(sram_accesses)
    return sram_accesses

layerNames = ["Conv3", "Conv3_Deep", "Bottleneck", "Pointwise", "Residual", "Depthwise-Separable"]
widths = ["5x80", "10x40", "20x20", "40x10", "80x5"]

cycles = []
utils = []
sram_accesses = []

for i in range(len(layerNames)):
    cycle, util = grabCycleUtil(paths[i], networks[i])
    cycles.append(cycle)
    utils.append(util)
    sram_accesses.append(grabSRAM(paths[i], networks[i]))

    print(cycles[i])
#print(sram_accesses[0])

# https://www.rapidtables.com/web/color/RGB_Color.html
ssize=100
ax = plt.subplot()
ax.scatter(sram_accesses[0], cycles[0], s=ssize, c=['#FF6666', '#FF0000', '#990000', '#FF0000', '#FF6666'], marker='o') # red circles
ax.scatter(sram_accesses[1], cycles[1], s=ssize, c=['#FFB266', '#FF8000', '#CC6600', '#FF8000', '#FFB266'], marker='s') # orange squares
#ax.scatter(sram_accesses[1], cycles[1], s=ssize, c=['#FFFF99', '#FFFF33', '#CCCC00', '#FFFF33', '#FFFF99'], marker='s') # yellow squares
ax.scatter(sram_accesses[2], cycles[2], s=ssize, c=['#33FF33', '#00CC00', '#006600', '#00CC00', '#33FF33'], marker='P') # green plus
ax.scatter(sram_accesses[3], cycles[3], s=ssize, c=['#A0A0A0', '#606060', '#000000', '#606060', '#A0A0A0'], marker='*') # grey stars
ax.scatter(sram_accesses[4], cycles[4], s=ssize, c=['#FF66FF', '#FF00FF', '#990099', '#FF00FF', '#FF66FF'], marker='X') # purple X
ax.scatter(sram_accesses[5], cycles[5], s=ssize, c=['#66B2FF', '#0080FF', '#0000FF', '#0080FF', '#66B2FF'], marker='D') # blue diamonds
ax.set_xlabel('Number GLB accesses')
ax.set_ylabel('Latency')
ax.legend(layerNames)
plt.xlim(left=0)
plt.grid()
plt.show()

"""
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
    util_diffs.append(utils[util_maxes_idx[layer], layer] - utils[4, layer])
util_diffs = [float(i)/max(util_diffs) for i in util_diffs]

#ax = plt.subplot()
#ax.plot(x, util_diffs, marker='o')
#ax.set_title("util_diffs")
#plt.grid(axis='x')
#plt.show()

# --- sram_diffs ---
sram_diffs = []
for layer in range(np.shape(utils)[1]):
    sram_diffs.append(sram_accesses[4, layer] - sram_accesses[sram_mins_idx[layer], layer])
sram_diffs = [float(i)/max(sram_diffs) for i in sram_diffs]

#ax = plt.subplot()
#ax.plot(x, sram_diffs, marker='o')
#ax.set_title("sram_diffs")
#plt.grid(axis='x')
#plt.show()

# --- cycle_diffs ---
cycle_diffs = []
for layer in range(np.shape(utils)[1]):
    cycle_diffs.append(cycles[4, layer] - cycles[cycle_mins_idx[layer], layer])
cycle_diffs = [float(i)/max(cycle_diffs) for i in cycle_diffs]

ax = plt.subplot()
#ax.plot(x, util_diffs, marker='o')
ax.plot(x, sram_diffs, marker='o')
ax.plot(x, cycle_diffs, marker='o')
ax.set_title("Diffs")
ax.legend(["sram_diffs", "cycle_diffs"])
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
"""
