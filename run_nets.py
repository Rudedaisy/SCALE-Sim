import trace_gen_wrapper as tg
import loadData as ld
import os
import subprocess
from util import latOptimizeShape

#COEFF_IDX_FILE = 'topologies/conv_nets/VGG19_agr_9248.h5' # used for our VGG19 experiment
#COEFF_IDX_FILE = 'topologies/conv_nets/VGG19_CIFAR100_sparse_weight.h5'
#COEFF_IDX_FILE = 'topologies/conv_nets/VGG19_ImageNet_sparse_weight.h5'
#COEFF_IDX_FILE = 'topologies/conv_nets/VGG16_sparse_weight_BAL_9227.pt'
#COEFF_IDX_FILE = 'topologies/conv_nets/AlexNet_sparse_weight.h5'
#COEFF_IDX_FILE = 'topologies/conv_nets/ResNet50_sparse_weight.h5'
#COEFF_IDX_FILE = "topologies/conv_nets/VGG16_ImageNet_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/ResNet18_9272_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/ResNet56_9248_shrinked_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/ResNet56_XXXX_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/VGG16_9370_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/ResNet18_9430_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/VGG16_9342_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/MobileNetV2_9335_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/ResNet50_sparse_weight_7604.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/ResNet152_9518_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/VGG16_ImageNet_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/MobileNetV2_ImageNet_sparse_weight.h5"
#COEFF_IDX_FILE = "topologies/conv_nets/ResNet50_sparse_weight_7376.h5" # experimental

#COEFF_IDX_FILE = "topologies/conv_nets/MBNETV2_s1.0_46.16_56.36_71.5.h5"
COEFF_IDX_FILE = "topologies/conv_nets/MBNETV2_s2.0_62.23_75.74_XXX.h5"

def run_net( ifmap_sram_size=1,
             filter_sram_size=1,
             ofmap_sram_size=1,
             wa_sram_size=1,
             array_h=32,
             array_w=32,
             wa_array_h=32,
             wa_array_scaleout=32,
             add_tree_leaves = 512,
             data_flow = 'os',
             topology_file = './topologies/yolo_v2.csv',
             net_name='yolo_v2',
             offset_list = [0, 10000000, 20000000, 30000000],
             PENNI=False,
             WAComp=False,
             num_bases=5,
             alternateOsWs=False,
             dynamicShape=True,
             dy_dim_list=[5, 10, 20, 40, 80]
            ):

    # coeff_ptrs format: [layer][output_channel] = [basis*num_input_channels + input_channel]
    coeff_ptrs = ld.loadCoeffIdx(COEFF_IDX_FILE)

    ifmap_sram_size *= 1024
    filter_sram_size *= 1024
    ofmap_sram_size *= 1024

    #fname = net_name + ".csv"
    param_file = open(topology_file, 'r')

    fname = net_name + "_avg_bw.csv"
    bw = open(fname, 'w')

    f2name = net_name + "_max_bw.csv"
    maxbw = open(f2name, 'w')

    f3name = net_name + "_cycles.csv"
    cycl = open(f3name, 'w')

    f4name = net_name + "_detail.csv"
    detail = open(f4name, 'w')

    bw.write("IFMAP SRAM Size,\tFilter SRAM Size,\tOFMAP SRAM Size,\tConv Layer Num,\tDRAM IFMAP Read BW,\tDRAM Filter Read BW,\tDRAM OFMAP Write BW,\tSRAM Read BW,\tSRAM OFMAP Write BW, \n")
    maxbw.write("IFMAP SRAM Size,\tFilter SRAM Size,\tOFMAP SRAM Size,\tConv Layer Num,\tMax DRAM IFMAP Read BW,\tMax DRAM Filter Read BW,\tMax DRAM OFMAP Write BW,\tMax SRAM Read BW,\tMax SRAM OFMAP Write BW,\n")
    cycl.write("Layer,\tCycles,\t% Utilization,\n")
    detailed_log = "Layer," +\
                 "\tDRAM_IFMAP_start,\tDRAM_IFMAP_stop,\tDRAM_IFMAP_bytes," + \
                 "\tDRAM_Filter_start,\tDRAM_Filter_stop,\tDRAM_Filter_bytes," + \
                 "\tDRAM_OFMAP_start,\tDRAM_OFMAP_stop,\tDRAM_OFMAP_bytes," + \
                 "\tSRAM_read_start,\tSRAM_read_stop,\tSRAM_read_bytes," +\
                 "\tSRAM_write_start,\tSRAM_write_stop,\tSRAM_write_bytes,\n"

    detail.write(detailed_log)


    first = True

    num_IFM_acc = []
    num_IFM_acc_fr = []
    num_filt_acc = []
    num_filt_acc_kr = []
    num_OFM_acc = []

    array_h_orig = array_h
    array_w_orig = array_w
    
    for row_idx, row in enumerate(param_file, start=-1):
        if first:
            first = False
            continue

        array_h = array_h_orig
        array_w = array_w_orig
        
        num_IFM_acc_layer = 0
        num_IFM_acc_fr_layer = 0
        num_filt_acc_layer = 0
        num_filt_acc_kr_layer = 0
        num_OFM_acc_layer = 0
            
        elems = row.strip().split(',')
        #print(len(elems))
        
        # Do not continue if incomplete line
        if len(elems) < 9:
            continue

        name = elems[0]
        print("")
        print("Commencing run for " + name)

        ifmap_h = int(elems[1])
        ifmap_w = int(elems[2])

        filt_h = int(elems[3])
        filt_w = int(elems[4])

        num_channels = int(elems[5])
        num_filters = int(elems[6])

        strides = int(elems[7])

        #print(str(len(elems)) + "~~~~~~~")
        print(elems)
        # Optional field: number of basis kernels
        if len(elems) > 9:
            num_bases = int(elems[8])

        ifmap_base  = offset_list[0]
        filter_base = offset_list[1]
        ofmap_base  = offset_list[2]
        wa_base = offset_list[3]

        bw_log = str(ifmap_sram_size) +",\t" + str(filter_sram_size) + ",\t" + str(ofmap_sram_size) + ",\t" + name + ",\t"
        max_bw_log = bw_log
        detailed_log = name + ",\t"

        # Added by Ed: in PENNI mode, set num_groups = num_channels if it's marked as Depthwise Seperable Conv (DSC)
        # if marked as Weighted Accumulate (WA), collect traces for the adder tree instead
        num_groups = 1
        DSC = False
        if PENNI and ("DSC" in name):
            num_groups = num_channels
            DSC = True
        if PENNI and WAComp and ("WA" in name):
            l_idx = int(name[2:]) - 1
            coeff_ptrs_layer = coeff_ptrs[l_idx] #coeff_ptrs[row_idx // 2]
            if dynamicShape:
                # determine best array shape here
                array_h, array_w = latOptimizeShape(ifmap_h, ifmap_w, filt_h, filt_w, num_channels, num_filters, strides, dy_dim_list, False, num_bases, True, coeff_ptrs_layer, l_idx, COEFF_IDX_FILE)            
            sq_ptrs = ld.squeezeCoeffIdx(coeff_ptrs_layer, array_w, num_bases, l_idx, COEFF_IDX_FILE, False, False, 1)

            clk = 0
            for chunk_idx, chunk in enumerate(sq_ptrs):
                #print("Chunk {} out of {}. Length of chunk: {}".format(chunk_idx, len(sq_ptrs), len(chunk)))
                if len(chunk) == 0: # and chunk_idx+1 < len(sq_ptrs):
                    print("Length of next chunks:")
                    for ci in range(chunk_idx, len(sq_ptrs)):
                        print("    {}: {}".format(ci, len(sq_ptrs[ci])))
                    continue
                bw_str, detailed_str, util, clk_chunk, num_IFM_acc_chunk, num_IFM_acc_fr_chunk, num_filt_acc_chunk, num_filt_acc_kr_chunk, num_OFM_acc_chunk = \
                    tg.gen_all_traces(
                                array_h = array_h,
                                array_w = array_w,
                                ifmap_h = ifmap_h,
                                ifmap_w = ifmap_w,
                                filt_h = filt_h,
                                filt_w = filt_w,
                                num_channels = len(chunk),
                                chunk_coeffs = chunk,
                                num_groups = num_groups,
                                num_filt = array_w, # we make assumption here
                                strides = strides,
                                data_flow = data_flow,
                                word_size_bytes = 1,
                                filter_sram_size = filter_sram_size,
                                ifmap_sram_size = ifmap_sram_size,
                                ofmap_sram_size = ofmap_sram_size,
                                filt_base = filter_base,
                                ifmap_base = ifmap_base,
                                ofmap_base = ofmap_base,
                                sram_read_trace_file= net_name + "_" + name + "chunk" + str(chunk_idx) + "_sram_read.csv",
                                sram_write_trace_file= net_name + "_" + name + "chunk" + str(chunk_idx) + "_sram_write.csv",
                                dram_filter_trace_file=net_name + "_" + name + "chunk" + str(chunk_idx) + "_dram_filter_read.csv",
                                dram_ifmap_trace_file= net_name + "_" + name + "chunk" + str(chunk_idx) + "_dram_ifmap_read.csv",
                                dram_ofmap_trace_file= net_name + "_" + name + "chunk" + str(chunk_idx) + "_dram_ofmap_write.csv",
                                PENNI=PENNI
                                )
                clk += int(clk_chunk)
                num_IFM_acc_layer += num_IFM_acc_chunk
                num_IFM_acc_fr_layer += num_IFM_acc_fr_chunk
                num_filt_acc_layer += num_filt_acc_chunk
                num_filt_acc_kr_layer += num_filt_acc_kr_chunk
                num_OFM_acc_layer += num_OFM_acc_chunk
                
                bw_log += bw_str
                bw.write(bw_log + "\n")

                detailed_log += detailed_str
                detail.write(detailed_log + "\n")

                max_bw_log += tg.gen_max_bw_numbers(
                                sram_read_trace_file = net_name + "_" + name + "chunk" + str(chunk_idx) + "_sram_read.csv",
                                sram_write_trace_file= net_name + "_" + name + "chunk" + str(chunk_idx) + "_sram_write.csv",
                                dram_filter_trace_file=net_name + "_" + name + "chunk" + str(chunk_idx) + "_dram_filter_read.csv",
                                dram_ifmap_trace_file= net_name + "_" + name + "chunk" + str(chunk_idx) + "_dram_ifmap_read.csv",
                                dram_ofmap_trace_file= net_name + "_" + name + "chunk" + str(chunk_idx) + "_dram_ofmap_write.csv"
                                )

                maxbw.write(max_bw_log + "\n")
            clk = str(clk)
            
        else:
            if dynamicShape:
                # determine best array shape here
                array_h, array_w = latOptimizeShape(ifmap_h, ifmap_w, filt_h, filt_w, num_channels, num_filters, strides, dy_dim_list, DSC, num_bases)
            
            bw_str, detailed_str, util, clk, num_IFM_acc_layer, num_IFM_acc_fr_layer, num_filt_acc_layer, num_filt_acc_kr_layer, num_OFM_acc_layer =  \
            tg.gen_all_traces(  array_h = array_h,
                                array_w = array_w,
                                ifmap_h = ifmap_h,
                                ifmap_w = ifmap_w,
                                filt_h = filt_h,
                                filt_w = filt_w,
                                num_channels = num_channels,
                                num_groups = num_groups,
                                num_filt = num_filters,
                                strides = strides,
                                data_flow = data_flow,
                                word_size_bytes = 1,
                                filter_sram_size = filter_sram_size,
                                ifmap_sram_size = ifmap_sram_size,
                                ofmap_sram_size = ofmap_sram_size,
                                filt_base = filter_base,
                                ifmap_base = ifmap_base,
                                ofmap_base = ofmap_base,
                                sram_read_trace_file= net_name + "_" + name + "_sram_read.csv",
                                sram_write_trace_file= net_name + "_" + name + "_sram_write.csv",
                                dram_filter_trace_file=net_name + "_" + name + "_dram_filter_read.csv",
                                dram_ifmap_trace_file= net_name + "_" + name + "_dram_ifmap_read.csv",
                                dram_ofmap_trace_file= net_name + "_" + name + "_dram_ofmap_write.csv",
                                PENNI = PENNI,
                                DSC = DSC,
                                alternateOsWs=alternateOsWs,
                                #dynamicShape=dynamicShape,
                                #dy_array_h=dy_array_h,
                                #dy_array_w=dy_array_w,
                                num_bases = num_bases
                            )

            bw_log += bw_str
            bw.write(bw_log + "\n")

            detailed_log += detailed_str
            detail.write(detailed_log + "\n")
            #"""
            max_bw_log += tg.gen_max_bw_numbers(
                                sram_read_trace_file = net_name + "_" + name + "_sram_read.csv",
                                sram_write_trace_file= net_name + "_" + name + "_sram_write.csv",
                                dram_filter_trace_file=net_name + "_" + name + "_dram_filter_read.csv",
                                dram_ifmap_trace_file= net_name + "_" + name + "_dram_ifmap_read.csv",
                                dram_ofmap_trace_file= net_name + "_" + name + "_dram_ofmap_write.csv"
                                )

            maxbw.write(max_bw_log + "\n")
            #""" ############## REMOVE THIS
        # Anand: This is not needed, sram_traffic() returns this
        #last_line = subprocess.check_output(["tail","-1", net_name + "_" + name + "_sram_write.csv"] )
        #clk = str(last_line).split(',')[0]
        #clk = str(clk).split("'")[1]

        #if not ("FC" in name): ###### Other accelerator works only concerned with CONV layers
        num_IFM_acc.append( num_IFM_acc_layer )
        num_IFM_acc_fr.append( num_IFM_acc_fr_layer )
        num_filt_acc.append( num_filt_acc_layer )
        num_filt_acc_kr.append( num_filt_acc_kr_layer )
        num_OFM_acc.append( num_OFM_acc_layer )

        util_str = str(util)
        line = name + ",\t" + clk +",\t" + util_str +",\n"
        cycl.write(line)

    print("---SRAM ACCESS COUNTS---")
    print("*FC Layers are NOT counted")
    print("IFM SRAM access (no fr): {}".format(sum(num_IFM_acc)))
    print("IFM SRAM access (with fr): {}".format(sum(num_IFM_acc_fr)))
    print("OFM SRAM access: {}".format(sum(num_OFM_acc)))
    print("Filt SRAM access (no kr): {}".format(sum(num_filt_acc)))
    print("Filt SRAM access (with kr): {}".format(sum(num_filt_acc_kr)))
    #print("Total SRAM reads (with kr): {}".format(num_IFM_acc + num_filt_acc + num_OFM_acc))
    #print("Total SRAM writes (with kr): {}".format(num_IFM_acc + num_filt_acc_kr + num_OFM_acc))

    bw.close()
    maxbw.close()
    cycl.close()
    param_file.close()

    return num_IFM_acc, num_IFM_acc_fr, num_OFM_acc, num_filt_acc, num_filt_acc_kr

#if __name__ == "__main__":
#    sweep_parameter_space_fast()    

