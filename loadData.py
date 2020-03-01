# File:    loadData.py
# Author:  Edward Hanson (eth20@duke.edu)

import torch
import numpy as np
import math

CHECK = 4

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

    """
    check = 5
    numBases = 5
    for idx, layer in enumerate(coeff_ptrs):
        tot_in = widths[idx] * numBases
        taken = [0 for _ in range(tot_in)]
        for out_channel in layer:
            for c in out_channel:
                taken[c] += 1
        print("Layer {} -- {} % of in_channels*B are used greater than {} times".format(idx, len([i for i in taken if i > check]) / tot_in, check))
    #"""
    return coeff_ptrs

# return squeezed list of array_w-wide array for a single layer
def squeezeCoeffIdx(coeff_ptrs_layer, array_w, numBases, layerIdx, pathName):
    data = torch.load(pathName)
    widths = [3] # first input channel width in VGG16
    widths.extend(data['width'])
    
    sq_ptrs = []
    num_fold = math.ceil(len(coeff_ptrs_layer) / array_w)
    max_len = numBases * widths[layerIdx]

    # internal counter just for information
    tot_pruned = 0
    
    for chunk in range(num_fold):
        num_pruned = 0
        sq_ptrs.append([])
        if (chunk+1) * array_w >= len(coeff_ptrs_layer):
            maxL = len(coeff_ptrs_layer)
        else:
            maxL = (chunk+1) * array_w
        for i in range(max_len):
            if any(i in sublist for sublist in coeff_ptrs_layer[(chunk * array_w):maxL]):
                sq_ptrs[len(sq_ptrs)-1].append(i)
            else:
                num_pruned += 1
                tot_pruned += 1
        #print("Number removed = {}, ratio = {}".format(num_pruned, num_pruned / max_len))
        
        """
        sq_ptrs.extend([[] for __ in range(array_w)])
        for i in range(max_len):
            # iterate through each possible coeff idx
            # if found, any corresponding out_channel will be appended the idx
            #           other channels will be a appended a false 0
            if (chunk+1) * array_w >= len(coeff_ptrs_layer):
                maxL = len(coeff_ptrs_layer)
            else:
                maxL = (chunk+1) * array_w
            if i in coeff_ptrs_layer[(chunk * array_w):maxL]:
                for w_idx in range(maxL):
                    if i in coeff_ptrs_layer[(chunk * array_w) + w_idx]:
                        sq_ptrs[len(sq_ptrs)-1][w_idx].append(i)
                    else:
                        sq_ptrs[len(sq_ptrs)-1][w_idx].append(0)
        """
    print("Number removed = {}, ratio = {}".format(tot_pruned, tot_pruned / (max_len * num_fold)))
    return sq_ptrs

coeff_ptrs = loadCoeffIdx('topologies/conv_nets/sparse_sample_weight.pt')

for i in range(len(coeff_ptrs)):
    coeff_ptrs_layer = coeff_ptrs[i]
    sq_ptrs = squeezeCoeffIdx(coeff_ptrs_layer, 32, 5, i, 'topologies/conv_nets/sparse_sample_weight.pt')
    print([len(chunk) for chunk in sq_ptrs])

#print("Original")
#print(coeff_ptrs_layer[0:32])
#print("Squeezed")
#print(sq_ptrs[0])

#coeff_ptrs = loadCoeffIdx('topologies/conv_nets/sparse_sample_weight.pt')
"""
# check if there are any common coeff_ptrs across outputs
# IMPORTANT: we assume there is NO SHARING so that results from DSC layer can be immediately piped to WA
#            stage with no storing in SRAM
print(coeff_ptrs[0])
for idx, layer in enumerate(coeff_ptrs):
    num_match = 0
    for out_channel in layer:
        for i1, c_ptr in enumerate(out_channel):
            for out_channel2 in layer:
                for i2, c_ptr2 in enumerate(out_channel2):
                    if c_ptr == c_ptr2 and i1 != i2:
                        num_match += 1
    print("Layer {} -- Number of repeated IFM ptrs: {}".format(idx, num_match))

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
