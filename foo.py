import numpy as np

path = "topologies/conv_nets/VGG16_ImageNet.csv"
f = open(path, 'r')

numParams = 0
maxSizeIFM = 0
whichIFM = ""
for idx, layer in enumerate(f):
    if idx < 5:
        continue
    elems = layer.strip().split(',')
    if len(elems) < 8:
        continue
    numParams += int(elems[3]) * int(elems[4]) * int(elems[5]) * int(elems[6])
    oldSizeIFM = maxSizeIFM
    maxSizeIFM = max(maxSizeIFM, int(elems[1]) * int(elems[2]) * int(elems[5]))
    if maxSizeIFM > oldSizeIFM:
        whichIFM = elems[0]
    
print("Number of params: {}".format(numParams))
print("Max IFM size: {} - {} bytes".format(whichIFM, maxSizeIFM))
