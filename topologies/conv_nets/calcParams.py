import os
import subprocess

topology_file = "VGG16_CIFAR10_AOFP.csv"

param_file = open(topology_file, 'r')
numParams = 0
first = True
for row_idx, row in enumerate(param_file, start=-1):
    if first:
        first = False
        continue
    elems = row.strip().split(',')
    numParams += (int(elems[3])*int(elems[4])*int(elems[5])*int(elems[6]))

print("Number of parameters: {}".format(numParams))
