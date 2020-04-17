# File:    loadData.py
# Author:  Edward Hanson (eth20@duke.edu)

import torch
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy

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

    print(widths)
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
def squeezeCoeffIdxOLD(coeff_ptrs_layer, array_w, numBases, layerIdx, pathName):
    data = torch.load(pathName)
    widths = [3] # first input channel width in VGG16
    widths.extend(data['width'])
    
    sq_ptrs = []
    num_fold = math.ceil(len(coeff_ptrs_layer) / array_w)
    max_len = numBases * widths[layerIdx]

    # internal counter just for information
    tot_pruned = 0
    nonzero_rows = 0
    
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
                nonzero_rows += 1
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
    #print("Number removed = {}, ratio = {}".format(tot_pruned, tot_pruned / (max_len * num_fold)))
    print("Score for layer {}: {}".format(layerIdx, tot_pruned / (max_len*num_fold)))
    return sq_ptrs

# return squeezed list of array_w-wide array for a single layer
def squeezeCoeffIdxOnce(coeff_ptrs_layer, chunks, array_w, in_len):
    # chunks: [out_channel_start, out_chanel_end+1]
    assert max(chunks) >= len(coeff_ptrs_layer)

    # keep track of pruning performance
    tot_pruned = 0
    tot_rounds = 0

    sq_ptrs = []
    for chunk_idx in range(len(chunks) - 1):
        assert array_w >= (chunks[chunk_idx + 1] - chunks[chunk_idx]), "Issue: {} vs {}".format(chunks[chunk_idx + 1], chunks[chunk_idx])

        chunk_ptrs = []
        for in_idx in range(in_len):
            if any(in_idx in sublist for sublist in coeff_ptrs_layer[chunks[chunk_idx]:chunks[chunk_idx + 1]]):
                chunk_ptrs.append(in_idx)
                tot_rounds += 1
            #else:
            #    tot_pruned += 1
        sq_ptrs.append(chunk_ptrs)

    #score = tot_pruned / len(chunks - 1)
    score = tot_rounds
    return sq_ptrs, score

# wrapper function to produce optimal chunks
def squeezeCoeffIdx(coeff_ptrs_layer, array_w, numBases, layerIdx, pathName, dynamicChunk=False, precompChunk=False, addChunk=0):
    if (not dynamicChunk) or layerIdx == 12:
        return squeezeCoeffIdxOLD(coeff_ptrs_layer, array_w, numBases, layerIdx, pathName)

    # IDEA: move fold and other precalculations out of wrapped function
    # make wrapped function simply squeeze chunks based on dynamic sizes of 'array_w'
    data = torch.load(pathName)
    widths = [3] # first input channel width in VGG16
    widths.extend(data['width'])
    
    num_fold = int(math.ceil(len(coeff_ptrs_layer) / array_w))
    in_len = numBases * widths[layerIdx]

    precomps = [[0, 23, 55],
                [0, 32, 64],
                [0, 32, 64, 96, 128],
                [0, 32, 64, 96, 128],
                [0, 32, 64, 96, 128, 160, 192, 224, 256],
                [0, 32, 64, 96, 128, 160, 192, 224, 256],
                [0, 32, 64, 77, 109, 141, 173, 205, 237],
                [0, 32, 64, 94, 126, 158],
                [0, 32, 62],
                [0, 16, 48],
                [0, 32, 36],
                [0, 26] # need chunk 13
                ]
    if precompChunk:
        if layerIdx != 12:
            chunks = precomps[layerIdx]
            return squeezeCoeffIdxOnce(coeff_ptrs_layer, chunks, array_w, in_len)[0]

    # initialize chunks list
    chunks = []
    numChunks = num_fold + addChunk
    for i in range(numChunks + 1):
        chunks.append(max(i, widths[layerIdx + 1] - (numChunks-i)*array_w))
    chunks[numChunks] = widths[layerIdx + 1]
    if numChunks > 1:
        chunks[numChunks-1] -= 1

    # progress bar to prevent going crazy
    print("Generating best chunk permutation")
    tot = array_w ** (numChunks)
    #pbar = tqdm(total=tot)

    # iterate through all partitions and find the best one
    sq_ptrs_best = []
    best_score = squeezeCoeffIdxOnce(coeff_ptrs_layer, precomps[layerIdx], array_w, in_len)[1] # in_len * len(chunks) + 1 # we want to minimize this score
    best_chunks = precomps[layerIdx] # []
    endPermute = False
    while not endPermute:
        #pbar.update(1)
        skipTest = False

        # helper variable to iterate through all possible permutations
        targ_chunk = numChunks - 1
        while True:
            if numChunks <= 1:
                endPermute = True
                break
            if chunks[targ_chunk] < widths[layerIdx + 1] - 1:
                chunks[targ_chunk] += 1
                if chunks[targ_chunk] - chunks[targ_chunk-1] > array_w:
                    break
            else:
                #chunks[targ_chunk] = chunks[targ_chunk - 1]
                targ_chunk -= 1
                if targ_chunk == 0:
                    #skipTest = True
                    endPermute = True
                    break
                continue
            if targ_chunk != numChunks - 1:
                for i in range(targ_chunk+1, numChunks):
                    chunks[i] = max(chunks[i-1] + 1, widths[layerIdx + 1] - (numChunks-i)*array_w)
            break

        # do not test impossible permutations
        for i in range(len(chunks)-1):
            if chunks[i+1] - chunks[i] > array_w:
                skipTest = True

        if not skipTest:
            #print(chunks)
            # test and find the best permutation
            curr_sq_ptrs, curr_score = squeezeCoeffIdxOnce(coeff_ptrs_layer, chunks, array_w, in_len)
            if curr_score < best_score:
                sq_ptrs_best = deepcopy(curr_sq_ptrs)
                best_score = curr_score
                best_chunks = deepcopy(chunks)
                print()
                print("Scores {} {}".format(best_score, in_len * len(chunks)))
                print("Current best chunks permutation: {}".format(best_chunks))
            print("{} -- score: [{}]".format(chunks, curr_score), end="\r")
    print("Scores {} {}".format(best_score, in_len * len(chunks)))

    #pbar.close()
    print("Best chunks permutation: {}".format(best_chunks))
    return sq_ptrs_best

        

coeff_ptrs = loadCoeffIdx('topologies/conv_nets/sparse_sample_weight.pt')
"""
for i in range(len(coeff_ptrs)):
    coeff_ptrs_layer = coeff_ptrs[i]
    sq_ptrs = squeezeCoeffIdx(coeff_ptrs_layer, 32, 5, i, 'topologies/conv_nets/sparse_sample_weight.pt')
    print([len(chunk) for chunk in sq_ptrs])
"""
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
