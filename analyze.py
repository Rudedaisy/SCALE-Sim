# Import matplotlib dependencies
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os.path
import math

peDim = 32
scaleOut = 32
CFLAG = 0

# base path | PENNI path
paths = ["outputs/VGG16_CIFAR10_32x32_os/", "outputs/VGG16_CIFAR10_32x32_os_PENNI/", "outputs/VGG16_CIFAR10_32x32_os_squeeze/", "outputs/VGG16_CIFAR10_32x32_os_PENNIv2/", "outputs/VGG16_CIFAR10_32x32_os_PENNIv3/"]
# cycles | avg_bandwidth
baseFileNames = ["VGG16_cycles.csv", "VGG16_avg_bw.csv"]
PENNIFileNames = ["VGG16_PENNIv1_cycles.csv", "VGG16_PENNIv1_avg_bw.csv"]
squeezeFileNames = ["VGG16_squeeze_cycles.csv", "VGG16_squeeze_avg_bw.csv"]
PENNI2FileNames = ["VGG16_PENNIv2_cycles.csv", "VGG16_PENNIv2_avg_bw.csv"]
files = []
for i in range(len(baseFileNames)):
    files.append(paths[0] + baseFileNames[i])
for i in range(len(PENNIFileNames)):
    files.append(paths[1] + PENNIFileNames[i])
for i in range(len(squeezeFileNames)):
    files.append(paths[2] + squeezeFileNames[i])
for i in range(len(PENNI2FileNames)):
    files.append(paths[3] + PENNI2FileNames[i])
for i in range(len(PENNI2FileNames)):
    files.append(paths[4] + PENNI2FileNames[i])

# base dram and sram bandwidths
baseDRAMRead = []
baseDRAMWrite = []
baseSRAMRead = []
baseSRAMWrite = []
f = open(files[1],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    baseDRAMRead.append(float(elems[4])+float(elems[5]))
    baseDRAMWrite.append(float(elems[6]))
    baseSRAMRead.append(float(elems[7]))
    baseSRAMWrite.append(float(elems[8]))
f.close()

# PENNIv1 dram and sram bandwidths
PDRAMRead = []
PDRAMWrite = []
PSRAMRead = []
PSRAMWrite = []
f = open(files[3],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    PDRAMRead.append(float(elems[4])+float(elems[5]))
    PDRAMWrite.append(float(elems[6]))
    PSRAMRead.append(float(elems[7]))
    PSRAMWrite.append(float(elems[8]))
f.close()

# squeeze dram and sram bandwidths
sDRAMRead = []
sDRAMWrite = []
sSRAMRead = []
sSRAMWrite = []
f = open(files[5],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    sDRAMRead.append(float(elems[4])+float(elems[5]))
    sDRAMWrite.append(float(elems[6]))
    sSRAMRead.append(float(elems[7]))
    sSRAMWrite.append(float(elems[8]))
f.close()

# PENNIv2 dram and sram bandwidths
P2DRAMRead = []
P2DRAMWrite = []
P2SRAMRead = []
P2SRAMWrite = []
f = open(files[7],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    P2DRAMRead.append(float(elems[4])+float(elems[5]))
    P2DRAMWrite.append(float(elems[6]))
    P2SRAMRead.append(float(elems[7]))
    P2SRAMWrite.append(float(elems[8]))
f.close()

# PENNIv3 dram and sram bandwidths
P3DRAMRead = []
P3DRAMWrite = []
P3SRAMRead = []
P3SRAMWrite = []
f = open(files[9],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    P3DRAMRead.append(float(elems[4])+float(elems[5]))
    P3DRAMWrite.append(float(elems[6]))
    P3SRAMRead.append(float(elems[7]))
    P3SRAMWrite.append(float(elems[8]))
f.close()
    
# base cycles and utilization
layerNames = []
baseCyles = []
baseUtilization = []
f = open(files[0],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    layerNames.append(elems[0])
    baseCyles.append(float(elems[1]))
    baseUtilization.append(float(elems[2]))
f.close()

# PENNI cycles and utilization
PENNICyles = []
PENNICombCyles = []
PENNIUtilization = []
f = open(files[2],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    PENNICyles.append(float(elems[1]))
    if "WA" in elems[0]:
        PENNICombCyles[len(PENNICombCyles) - 1] += float(elems[1])
    else:
        PENNICombCyles.append(float(elems[1]))
    
    PENNIUtilization.append(float(elems[2]))
f.close()

# baseline squeeze cycles and utilization
squeezeCyles = []
squeezeUtilization = []
f = open(files[4],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    squeezeCyles.append(float(elems[1]))
    squeezeUtilization.append(float(elems[2]))
f.close()

# squeeze + decouple cycles and utilization
PENNI2Cyles = []
PENNI2CombCyles = []
PENNI2Utilization = []
f = open(files[6],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    PENNI2Cyles.append(float(elems[1]))
    if "WA" in elems[0]:
        PENNI2CombCyles[len(PENNI2CombCyles) - 1] += float(elems[1])
    else:
        PENNI2CombCyles.append(float(elems[1]))

    PENNI2Utilization.append(float(elems[2]))
f.close()

# squeeze + decouple + adder tree cycles and utilization
PENNI3Cyles = []
PENNI3CombCyles = []
PENNI3Utilization = []
f = open(files[8],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    PENNI3Cyles.append(float(elems[1]))
    if "WA" in elems[0]:
        PENNI3CombCyles[len(PENNI3CombCyles) - 1] += float(elems[1])
    else:
        PENNI3CombCyles.append(float(elems[1]))

    PENNI3Utilization.append(float(elems[2]))
f.close()

# squeeze + decouple + smart acc cycles and utilization
coeff_count = [[196, 2238, 5862, 12052, 23169, 36870, 27716, 15665, 4530, 2666, 1315, 874, 2554], \
               [960,20480,40960,81920,163840,327680,327680,655360,13010720,13010720,13010720,13010720,13010720]]
coeff_count = coeff_count[CFLAG]
IFM_sizes = [32,32,16,16,8,8,8,4,4,4,2,2,2]
PENNI4Cyles = []
PENNI4CombCyles = []
PENNI4Utilization = []
f = open(files[8],'r')
for idx, layer in enumerate(f, start=-1):
    if idx == -1:
        continue
    elems = layer.strip().split(',')
    #PENNI4Cyles.append(float(elems[1]))
    if "WA" in elems[0]:
        cycle_estimate = coeff_count[idx // 2] * ((IFM_sizes[idx // 2]**2) / (peDim * scaleOut))
        cycle_estimate += peDim # drain the remaining results
        PENNI4CombCyles[len(PENNI4CombCyles) - 1] += cycle_estimate
        PENNI4Cyles.append(cycle_estimate)
    else:
        PENNI4CombCyles.append(float(elems[1]))
        PENNI4Cyles.append(float(elems[1]))
        

    #PENNI4Utilization.append(float(elems[2]))
f.close()

# --------------------
# --- Print Results --
# --------------------

print("Total latency speedup")
print("PENNIv1: {}".format(np.sum(baseCyles) / np.sum(PENNICombCyles)))
print("Squeeze: {}".format(np.sum(baseCyles) / np.sum(squeezeCyles)))
print("PENNIv2: {}".format(np.sum(baseCyles) / np.sum(PENNI2CombCyles)))
print("PENNIv3: {}".format(np.sum(baseCyles) / np.sum(PENNI3CombCyles)))
print("PENNIv4: {}".format(np.sum(baseCyles) / np.sum(PENNI4CombCyles)))
print("Throughput speedup")
print("PENNIv1: {}".format(max(baseCyles) / max(PENNICombCyles)))
print("Squeeze: {}".format(max(baseCyles) / max(squeezeCyles)))
print("PENNIv2: {}".format(max(baseCyles) / max(PENNI2CombCyles)))
print("PENNIv3: {}".format(max(baseCyles) / max(PENNI3CombCyles)))
print("PENNIv4: {}".format(max(baseCyles) / max(PENNI4CombCyles)))
print("Peak DRAM Read BW")
print("Baseline: {}".format(max(baseDRAMRead)))
print("PENNIv1: {}".format(max(PDRAMRead)))
print("Squeeze: {}".format(max(sDRAMRead)))
print("PENNIv2: {}".format(max(P2DRAMRead)))
print("PENNIv3: {}".format(max(P3DRAMRead)))
print("Peak DRAM Write BW")
print("Baseline: {}".format(max(baseDRAMWrite)))
print("PENNIv1: {}".format(max(PDRAMWrite)))
print("Squeeze: {}".format(max(sDRAMWrite)))
print("PENNIv2: {}".format(max(P2DRAMWrite)))
print("PENNIv3: {}".format(max(P3DRAMWrite)))
print("Peak SRAM Read BW")
print("Baseline: {}".format(max(baseSRAMRead)))
print("PENNIv1: {}".format(max(PSRAMRead)))
print("Squeeze: {}".format(max(sSRAMRead)))
print("PENNIv2: {}".format(max(P2SRAMRead)))
print("PENNIv3: {}".format(max(P3SRAMRead)))
print("Peak SRAM Write BW")
print("Baseline: {}".format(max(baseSRAMWrite)))
print("PENNIv1: {}".format(max(PSRAMWrite)))
print("Squeeze: {}".format(max(sSRAMWrite)))
print("PENNIv2: {}".format(max(P2SRAMWrite)))
print("PENNIv3: {}".format(max(P3SRAMWrite)))

# utility function to align PENNI and baseline data
def align(data):
    newData = []
    for idx, i in enumerate(data):
        if idx <= 12:
            newData.append(i)
        newData.append(i)
    return newData

# --------------------
# --- Plot Results ---
# --------------------

def plotBW():
    # DRAM Read BW
    fig, ax = plt.subplots()
    ax.plot(range(len(PDRAMRead)), align(baseDRAMRead))
    ax.plot(range(len(PDRAMRead)), PDRAMRead)
    ax.plot(range(len(PDRAMRead)), align(sDRAMRead))
    ax.plot(range(len(PDRAMRead)), P2DRAMRead)
    ax.plot(range(len(PDRAMRead)), P3DRAMRead)
    ax.set_xlabel('Layer', fontsize = 'large')
    ax.set_ylabel('Bandwidth (B/clk)', fontsize = 'large')
    ax.set_title('VGG16 DRAM Read Bandwidths')
    ax.legend(['Baseline','Decouple','Squeeze','Sq+Dc','Sq+Dc+Tree'], loc=2)
    ax.grid(True)
    plt.show()
    
    # DRAM Write BW
    fig, ax = plt.subplots()
    ax.plot(range(len(PDRAMRead)), align(baseDRAMWrite))
    ax.plot(range(len(PDRAMRead)), PDRAMWrite)
    ax.plot(range(len(PDRAMRead)), align(sDRAMWrite))
    ax.plot(range(len(PDRAMRead)), P2DRAMWrite)
    ax.plot(range(len(PDRAMRead)), P3DRAMWrite)
    ax.set_xlabel('Layer', fontsize = 'large')
    ax.set_ylabel('Bandwidth (B/clk)', fontsize = 'large')
    ax.set_title('VGG16 DRAM Write Bandwidths')
    ax.legend(['Baseline','Decouple','Squeeze','Sq+Dc','Sq+Dc+Tree'], loc=1)
    ax.grid(True)
    plt.show()

def plotCycleUtil():
    # raw cycle count
    fig, ax = plt.subplots()
    ax.plot(range(len(layerNames)), baseCyles)
    ax.plot(range(len(layerNames)), PENNICombCyles)
    ax.plot(range(len(layerNames)), squeezeCyles)
    ax.plot(range(len(layerNames)), PENNI2CombCyles)
    ax.plot(range(len(layerNames)), PENNI3CombCyles)
    ax.plot(range(len(layerNames)), PENNI4CombCyles)
    ax.set_xlabel('Layer', fontsize = 'large')
    ax.set_ylabel('Cycle Count', fontsize = 'large')
    ax.set_title('VGG16 Cycle Count')
    ax.legend(['Baseline','Decouple','Squeeze','Sq+Dc','Sq+Dc+Tree','PENNI'], loc=2)
    ax.grid(True)
    #plt.xlim(80,100)
    #plt.ylim(0.65,0.82)
    plt.show()
    
    # speedup
    fig, ax = plt.subplots()
    ax.plot(range(len(layerNames)), np.divide(baseCyles,baseCyles))
    ax.plot(range(len(layerNames)), np.divide(baseCyles,PENNICombCyles))
    ax.plot(range(len(layerNames)), np.divide(baseCyles,squeezeCyles))
    ax.plot(range(len(layerNames)), np.divide(baseCyles,PENNI2CombCyles))
    ax.plot(range(len(layerNames)), np.divide(baseCyles,PENNI3CombCyles))
    ax.plot(range(len(layerNames)), np.divide(baseCyles,PENNI4CombCyles))
    ax.set_xlabel('Layer', fontsize = 'large')
    ax.set_ylabel('Speedup', fontsize = 'large')
    ax.set_title('VGG16 Speedup')
    ax.legend(['Baseline','Decouple','Squeeze','Sq+Dc','Sq+Dc+Tree','PENNI'], loc=2)
    ax.grid(True)
    plt.yscale("log")
    plt.show()
    
    # PENNI methods cycle count
    fig, ax = plt.subplots()
    ax.plot(range(len(PENNICyles)), PENNICyles)
    ax.plot(range(len(PENNI2Cyles)), PENNI2Cyles)
    ax.plot(range(len(PENNI3Cyles)), PENNI3Cyles)
    ax.plot(range(len(PENNI4Cyles)), PENNI4Cyles)
    ax.set_xlabel('Layer', fontsize = 'large')
    ax.set_ylabel('Cycle Count', fontsize = 'large')
    ax.set_title('Decoupled Layers Cycle Count')
    ax.legend(['Decouple','Sq+Dc','Sq+Dc+Tree','PENNI'], loc=2)
    ax.grid(True)
    plt.show()
    
    # PENNI methods utilization
    fig, ax = plt.subplots()
    ax.plot(range(len(baseCyles)), baseUtilization)
    ax.plot(range(len(PENNICyles)), PENNIUtilization)
    ax.plot(range(len(PENNI2Cyles)), PENNI2Utilization)
    ax.plot(range(len(PENNI3Cyles)), PENNI3Utilization)
    ax.set_xlabel('Layer', fontsize = 'large')
    ax.set_ylabel('Percent utilization', fontsize = 'large')
    ax.set_title('Decoupled Layers Utilization')
    ax.legend(['baseline','Decouple','Sq+Dc','Sq+Dc+Tree'], loc=3)
    ax.grid(True)
    plt.show()

plotCycleUtil()
plotBW
