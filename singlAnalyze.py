import numpy as np

#path = "outputs/VGG19_ImageNet_32x8_os_PENNIv7/"
#path = "outputs/VGG19_ImageNet_32x8/"
#path = "outputs/VGG19_ImageNet_16x16_os_base_pca/"
#path = "outputs/VGG19_ImageNet_16x16_os_baseline/"
#path = "outputs/VGG16_ImageNet_64x8_os_PENNIv7/"
#path = "outputs/VGG16_CIFAR10_dynamic_os_PENNIv7/"
#path = "outputs/VGG16_CIFAR10_20x20_os_PENNIv7/"
#path = "outputs/VGG16_ImageNet_23x23_os_baseline/"
#path = "outputs/ResNet50_ImageNet_16x16_os_baseline/"
#path = "outputs/ResNet50_ImageNet_dynamic_os_PENNIv7/" #_CoCo/"
#path = "outputs/ResNet50_ImageNet_64x8_os_PENNIv7/"
#path = "outputs/ResNet18_CIFAR10_dynamic_os_PENNIv3/"
#path = "outputs/ResNet18_CIFAR10_16x16_os_SS/"
#path = "outputs/conv11_32_16_PENNIv7/"
#path = "outputs/ResNet18_CIFAR10_dynamic_os_PENNIv7/"
#path = "outputs/MobileNetV2_CIFAR10_dynamic_os_PENNIv7/"
#path = "outputs/ResNet152_CIFAR10_dynamic_os_PENNIv7/"
#path = "outputs/VGG16_ImageNet_dynamic_os_PENNIv7/"
#path = "outputs/ResNet50_ImageNet_dynamic_os_PENNIv7/"
path = "outputs/MobileNetV2_ImageNet_dynamic_PENNIv7/"
#path = "outputs/ResNet50_ImageNet_dynamic_os_PENNIv7_experimental/" # experimental

#path = "outputs/ResNet50_ImageNet_dynamic_os_baseline/"
#path = "outputs/ResNet50_ImageNet_20x20_os_PENNIv7/"
#path = "outputs/ResNet50_ImageNet_20x20_os_PENNIv7_noCompression/"
#path = "outputs/ResNet50_ImagNet_20x20_os_baseline/"

#path = "outputs/ResNet18_CIFAR10_dynamic_100_os_PENNIv7/"
#path = "outputs/ResNet18_CIFAR10_80x80_os_PENNIv7/"

modelName = "MobileNetV2_ImageNet"
appendix = ["_cycles.csv", "_avg_bw.csv", "_max_bw.csv"]

########
# Params to change
NUM_FC = 1
REMOVE_FC = True
clk_freq = 300000000.0 #300000000.0
#######

Cycles = []
CombCycles = []
Utilization = []
f = open(path + modelName + appendix[0],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    Cycles.append(float(elems[1]))
    if "WA" in elems[0]:
        CombCycles[len(CombCycles) - 1] += float(elems[1])
    else:
        CombCycles.append(float(elems[1]))

    Utilization.append(float(elems[2]))
f.close()

DRAMIFMRead = []
DRAMFiltRead = []
DRAMWrite = []
SRAMRead = []
SRAMWrite = []
f = open(path + modelName + appendix[1],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    DRAMIFMRead.append(float(elems[4]))
    DRAMFiltRead.append(float(elems[5]))
    DRAMWrite.append(float(elems[6]))
    SRAMRead.append(float(elems[7]))
    SRAMWrite.append(float(elems[8]))
f.close()

DRAMMIFMRead = []
DRAMFiltRead = []
DRAMMWrite = []
SRAMMRead = []
SRAMMWrite = []
f = open(path + modelName + appendix[2],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    DRAMMIFMRead.append(float(elems[4]))
    DRAMFiltRead.append(float(elems[5]))
    DRAMMWrite.append(float(elems[6]))
    SRAMMRead.append(float(elems[7]))
    SRAMMWrite.append(float(elems[8]))
f.close()


if REMOVE_FC:
    print("Total cycle count: {}".format(np.sum(Cycles[:-1 * NUM_FC])))
    print("Latency (s): {}".format(np.sum(Cycles[:-1 * NUM_FC]) / clk_freq))
else:
    print("Total cycle count: {}".format(np.sum(Cycles)))
    print("Latency (s): {}".format(np.sum(Cycles) / clk_freq))
#"""
#print("Max DRAM read: {} MB/sec".format(np.max(DRAMMRead[:-1 * NUM_FC])*clk_freq / 1000000.0))
print("Max DRAM write: {} MB/sec".format(np.max(DRAMMWrite[:-1 * NUM_FC])*clk_freq / 1000000.0))
print("Max SRAM read: {} MB/sec".format(np.max(SRAMMRead[:-1 * NUM_FC])*clk_freq / 1000000.0))
print("MAX SRAM write: {} MB/sec".format(np.max(SRAMMWrite[:-1 * NUM_FC])*clk_freq / 1000000.0))
#"""

# Calculate layer-wise number of DRAM accesses
countDRAMIFMRead = [a*b for a,b in zip(CombCycles[:-1 * NUM_FC], DRAMIFMRead[:-1 * NUM_FC])]
countDRAMFiltRead = [a*b for a,b in zip(CombCycles[:-1 * NUM_FC], DRAMFiltRead[:-1 * NUM_FC])]
countDRAMWrite = [a*b for a,b in zip(CombCycles[:-1 * NUM_FC], DRAMWrite[:-1 * NUM_FC])]
sumDRAMIFMRead = np.sum(countDRAMIFMRead[:-1 * NUM_FC])
sumDRAMFiltRead = np.sum(countDRAMFiltRead[:-1 * NUM_FC])
sumDRAMWrite = np.sum(countDRAMWrite[:-1 * NUM_FC])
ratioDRAMIFMRead = [a/sumDRAMIFMRead for a in countDRAMIFMRead]
ratioDRAMFiltRead = [a/sumDRAMFiltRead for a in countDRAMFiltRead]
ratioDRAMWrite = [a/sumDRAMWrite for a in countDRAMWrite]
print("DRAM IFM read count: {}".format(np.sum(countDRAMIFMRead)))
print("DRAM Filt read count: {}".format(np.sum(countDRAMFiltRead)))
print("DRAM write count: {}".format(np.sum(countDRAMWrite)))

# Calculate layer-wise number of SRAM accesses
#for idx, layer_cycles in enumerate(CombCycles):
"""
countSRAMRead = [a*b for a,b in zip(CombCycles[:-1 * NUM_FC], SRAMRead[:-1 * NUM_FC])]
countSRAMWrite = [a*b for a,b in zip(CombCycles[:-1 * NUM_FC], SRAMWrite[:-1 * NUM_FC])]
sumSRAMRead = np.sum(countSRAMRead[:-1 * NUM_FC])
sumSRAMWrite = np.sum(countSRAMWrite[:-1 * NUM_FC])
ratioSRAMRead = [a/sumSRAMRead for a in countSRAMRead]
ratioSRAMWrite = [a/sumSRAMWrite for a in countSRAMWrite]
print("SRAM read count: {}".format(np.sum(countSRAMRead)))
print("SRAM write count: {}".format(np.sum(countSRAMWrite)))
print("SRAM read freqs")
print([round(num, 2) for num in ratioSRAMRead])
print("SRAM write freqs")
print([round(num, 2) for num in ratioSRAMWrite])
#"""
