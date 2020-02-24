# File:    loadData.py
# Author:  Edward Hanson (eth20@duke.edu)

import torch
import numpy as np

def loadCoeffIdx(pathName):
    # Intermediate feature map for layer 'l' format expected: [h][w][bases][input_channel]
    # [layer][output_channel] = [basis*num_input_channels + input_channel]
    data = torch.load(pathName)
    widths = [3] # first input channel width in VGG16
    widths.extend(data['width'])
    coeff_ptrs = []
    for layer_idx, layer in enumerate(data['indices']):
        coeff_ptrs.append([])
        coeff_ptrs[layer_idx] = [[] for _ in range(widths[layer_idx + 1])]
        for c in layer:
            coeff_ptrs[layer_idx][c[0] // widths[layer_idx]].append(c[1]*widths[layer_idx] + (c[0] % widths[layer_idx]))
        for out_channel in range(len(coeff_ptrs[layer_idx])):
            coeff_ptrs[layer_idx][out_channel].sort()
    return coeff_ptrs


"""
coeff_ptrs = loadCoeffIdx('topologies/conv_nets/sparse_sample_weight.pt')
print(coeff_ptrs)
#print(np.shape(coeff_ptrs))
tot_acc = 0
max_acc = 0
num_outs = 0
for idx, layer in enumerate(coeff_ptrs):
    l_tot_acc = 0
    for out_channel in layer:
        tot_acc += len(out_channel)
        l_tot_acc += len(out_channel)
        num_outs += 1
        if (len(out_channel) > max_acc):
            max_acc = len(out_channel)
    print("Layer {} -- NumCoefs: {}".format(idx, l_tot_acc))
avg_acc = tot_acc / num_outs
print("Avg number of accs: {}".format(avg_acc))
print("Max number of accs: {}".format(max_acc))
#"""
