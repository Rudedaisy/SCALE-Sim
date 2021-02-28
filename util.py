import math
import numpy as np
import loadData as ld

def cycleEstimate(arr_h=20,
                  arr_w=20,
                  ofmap_h=32,
                  ofmap_w=32,
                  filt_h=3,
                  filt_w=3,
                  num_channels=256,
                  num_filters=256,
                  isDSC=False, num_bases=5):
    """ Estimate relative latency for the given parameters """
    if not isDSC:
        numFeeds = filt_h*filt_w*num_channels
        h_folds = math.ceil(float(ofmap_h * ofmap_w) / arr_h)
        w_folds = math.ceil(float(num_filters) / arr_w)
    else:
        numFeeds = filt_h*filt_w
        h_folds = math.ceil(float(ofmap_h * ofmap_w) / arr_h)
        w_folds = math.ceil(float(num_filters) / num_bases)

    return int(numFeeds * h_folds * w_folds) # + arr_h) # -- uncomment "arr_h" for arrays larger than 20x20

def latOptimizeShape(ifmap_h=32,
                     ifmap_w=32,
                     filt_h=3,
                     filt_w=3,
                     num_channels=256,
                     num_filters=256,
                     stride=1,
                     dy_dim_list=[5, 10, 20, 40, 80],
                     isDSC=False, num_bases=5,
                     isWA=False, coeff_ptrs_layer=[], l_idx=0, COEFF_IDX_FILE=""):
    """ optimize array shape for LATENCY """

    # compute ofmap dims using the stride
    ofmap_h = math.ceil(float(ifmap_h - 2*(filt_h // 2)) / stride)
    ofmap_w = math.ceil(float(ifmap_w - 2*(filt_w // 2)) / stride)

    scores = []
    for h_idx in range(len(dy_dim_list)):
        arr_h = dy_dim_list[h_idx]
        arr_w = dy_dim_list[len(dy_dim_list)-1-h_idx]
        
        if not isWA:
            scores.append(cycleEstimate(arr_h, arr_w, ofmap_h, ofmap_w, filt_h, filt_w, num_channels, num_filters, isDSC, num_bases))
        else:
            scores.append(0)
            sq_ptrs = ld.squeezeCoeffIdx(coeff_ptrs_layer, arr_w, num_bases, l_idx, COEFF_IDX_FILE, False, False, 1)
            print([len(chunk) for chunk in sq_ptrs])
            for chunk in sq_ptrs:
                score = cycleEstimate(arr_h, arr_w, ofmap_h, ofmap_w, filt_h, filt_w, len(chunk), math.ceil(float(num_filters) / arr_w), False)
                scores[-1] += score 

    # determine best dimensions
    print(scores)
    best_idx = np.argmin(scores)
    while(True):
        if ((best_idx+1) < len(scores)) and (scores[best_idx+1] <= scores[best_idx]):
            if abs((float(len(scores))/2) - best_idx) > abs((float(len(scores))/2) - (best_idx+1)):
                best_idx += 1
                continue
        elif ((best_idx-1) >= 0) and (scores[best_idx-1] <= scores[best_idx]):
            if abs((float(len(scores))/2) - best_idx) >	abs((float(len(scores))/2) - (best_idx-1)):
                best_idx -= 1
                continue
        break

    best_h = dy_dim_list[best_idx]
    best_w = dy_dim_list[len(dy_dim_list)-1-best_idx]

    return best_h, best_w
    
def main():
    #h, w = latOptimizeShape(7, 7, 3, 3, 512, 512, 1)
    #h, w = latOptimizeShape(16, 16, 3, 3, 305, 125, 1, [5, 10, 20, 40, 80], True, 5)
    #print([h,w])
    #return
    
    COEFF_IDX_FILE = "topologies/conv_nets/VGG16_9370_sparse_weight.h5"
    coeff_ptrs = ld.loadCoeffIdx(COEFF_IDX_FILE)
    ifms = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
    chans = [15, 145, 305, 455, 625, 815, 585, 510, 210, 75, 105, 65, 55]
    filts = [29, 61, 61, 125, 163, 117, 102, 42, 15, 21, 13, 11, 78]
    for l_idx in range(13):
        coeff_ptrs_layer = coeff_ptrs[l_idx]
        h, w = latOptimizeShape(ifms[l_idx], ifms[l_idx], 1, 1, chans[l_idx], filts[l_idx], 1, [5, 10, 20, 40, 80], False, 5, True, coeff_ptrs_layer, l_idx, COEFF_IDX_FILE)
        print([h, w])
        print()

if __name__ == "__main__":
    main()
