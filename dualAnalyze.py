# layer-wise speedup of dynamic shape vs static shape

import numpy as np
import matplotlib.pyplot as plt

pathDynamic = "outputs/ResNet18_CIFAR10_dynamic_400_os_PENNIv7/"
pathStatic = "outputs/ResNet18_CIFAR10_20x20_os_PENNIv7/"

modelName = "ResNet18"
appendix = "_cycles.csv"

NUM_FC = 1
REMOVE_FC = True

dynamicCycles = []
dynamicCombCycles = []
f = open(pathDynamic + modelName + appendix,'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    if "FC" in elems[0]:
        continue
    dynamicCycles.append(float(elems[1]))
    if "WA" in elems[0]:
        dynamicCombCycles[len(dynamicCombCycles) - 1] += float(elems[1])
    else:
        dynamicCombCycles.append(float(elems[1]))
f.close()

staticCycles = []
staticCombCycles = []
f = open(pathStatic + modelName + appendix,'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    if "FC" in elems[0]:
        continue
    staticCycles.append(float(elems[1]))
    if "WA" in elems[0]:
        staticCombCycles[len(staticCombCycles) - 1] += float(elems[1])
    else:
        staticCombCycles.append(float(elems[1]))
f.close()

print(staticCycles)
print(dynamicCycles)

for i in range(len(staticCycles)):
    if float(staticCycles[i]) / dynamicCycles[i] < 1:
        dynamicCycles[i] = 1.0
    else:
        dynamicCycles[i] = float(staticCycles[i]) / dynamicCycles[i]

labels = ['SKC', 'WA']
    
def subcategorybar(X, vals, width=0.8):
    global fig, ax
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        ax.bar(_X - width/2. + i/float(n)*width, vals[i],
               width=width/float(n), align="edge", label=labels[i])
    plt.xticks(_X, X)

DSCidx = [0,2,4,6,8,10,12,15,17,19,21,24,26,28,30,33,35]
WAidx = [1,3,5,7,9,11,13,16,18,20,22,25,27,29,31,34,36]
ResIdx = [14,23,32]

DSCcycles = [dynamicCycles[i] for i in DSCidx]
WACycles = [dynamicCycles[i] for i in WAidx]

print(DSCcycles)
print(WACycles)


fig, ax = plt.subplots()
x = [1, 2,3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20]
subcategorybar(x, [DSCcycles, WACycles])
ax.legend()
plt.ylim(bottom=0.99)
plt.show()

Rescycles = [dynamicCycles[i] for i in ResIdx]
fig, ax = plt.subplots()
x = ['8','13','18']
ax.bar(x, Rescycles)
ax.legend(['Residual Connection'])
plt.show()

#----------------------------------

appendix = "_sram_accesses.csv"
dynamicSRAM = []
f = open(pathDynamic + modelName + appendix,'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    dynamicSRAM.append(int(elems[1]) + int(elems[2]) + int(elems[4]))
f.close()

staticSRAM = []
f = open(pathStatic + modelName + appendix,'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    staticSRAM.append(int(elems[0]) + int(elems[2]) + int(elems[3]))
f.close()

print(dynamicSRAM)
print(staticSRAM)

for i in range(len(staticSRAM)):
    if float(staticSRAM[i]) / dynamicSRAM[i] < 1:
        dynamicSRAM[i] = 1.0
    else:
        dynamicSRAM[i] = float(staticSRAM[i]) / dynamicSRAM[i]

SKCsram = [dynamicSRAM[i] for i in DSCidx]
WAsram = [dynamicSRAM[i] for i in WAidx]

print(SKCsram)
print(WAsram)

fig, ax = plt.subplots()
x = [1, 2,3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20]
subcategorybar(x, [SKCsram, WAsram])
ax.legend()
plt.ylim(bottom=0.99)
plt.show()

ResSRAM = [dynamicSRAM[i] for i in ResIdx]
fig, ax = plt.subplots()
x = ['8','13','18']
ax.bar(x, ResSRAM)
ax.legend(['Residual Connection'])
plt.ylim(bottom=0.99)
plt.show()
