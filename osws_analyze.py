
pathws = "outputs/ResNet18_CIFAR10_20x20_ws_baseline/"

modelName = "ResNet18_baseline"
appendix = ["_cycles.csv", "_avg_bw.csv", "_max_bw.csv"]

NUM_FC = 1
REMOVE_FC = True

wsCycles = []
f = open(pathws + modelName + appendix[0],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    wsCycles.append(float(elems[1]))
f.close()

pathos = "outputs/ResNet18_CIFAR10_20x20_os_baseline/"

osCycles = []
f = open(pathos + modelName + appendix[0],'r')
for idx, layer in enumerate(f):
    if idx == 0:
        continue
    elems = layer.strip().split(',')
    osCycles.append(float(elems[1]))
f.close()

print("os")
print(osCycles)
print("ws")
print(wsCycles)
