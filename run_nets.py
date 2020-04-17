import trace_gen_wrapper as tg
import loadData as ld
import os
import subprocess

COEFF_IDX_FILE = 'topologies/conv_nets/VGG16_sparse_weight_BAL_9227.pt'


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
             num_bases=5
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

    num_IFM_acc = 0
    num_filt_acc = 0
    num_filt_acc_kr = 0
    num_OFM_acc = 0
    
    for row_idx, row in enumerate(param_file, start=-1):
        if first:
            first = False
            continue

        num_IFM_acc_layer = 0
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
            coeff_ptrs_layer = coeff_ptrs[row_idx // 2]
            sq_ptrs = ld.squeezeCoeffIdx(coeff_ptrs_layer, array_w, num_bases, row_idx // 2, COEFF_IDX_FILE, False, False, 1)
            

            clk = 0
            for chunk_idx, chunk in enumerate(sq_ptrs):
                bw_str, detailed_str, util, clk_chunk, num_IFM_acc_chunk, num_filt_acc_chunk, num_filt_acc_kr_chunk, num_OFM_acc_chunk = \
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
                                )
                clk += int(clk_chunk)
                num_IFM_acc_layer += num_IFM_acc_chunk
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
            bw_str, detailed_str, util, clk, num_IFM_acc_layer, num_filt_acc_layer, num_filt_acc_kr_layer, num_OFM_acc_layer =  \
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
                                DSC = DSC
                            )

            bw_log += bw_str
            bw.write(bw_log + "\n")

            detailed_log += detailed_str
            detail.write(detailed_log + "\n")

            max_bw_log += tg.gen_max_bw_numbers(
                                sram_read_trace_file = net_name + "_" + name + "_sram_read.csv",
                                sram_write_trace_file= net_name + "_" + name + "_sram_write.csv",
                                dram_filter_trace_file=net_name + "_" + name + "_dram_filter_read.csv",
                                dram_ifmap_trace_file= net_name + "_" + name + "_dram_ifmap_read.csv",
                                dram_ofmap_trace_file= net_name + "_" + name + "_dram_ofmap_write.csv"
                                )

            maxbw.write(max_bw_log + "\n")

        # Anand: This is not needed, sram_traffic() returns this
        #last_line = subprocess.check_output(["tail","-1", net_name + "_" + name + "_sram_write.csv"] )
        #clk = str(last_line).split(',')[0]
        #clk = str(clk).split("'")[1]

        if DSC:#True:
            num_IFM_acc += num_IFM_acc_layer
            num_filt_acc += num_filt_acc_layer
            num_filt_acc_kr += num_filt_acc_kr_layer
            num_OFM_acc += num_OFM_acc_layer

        util_str = str(util)
        line = name + ",\t" + clk +",\t" + util_str +",\n"
        cycl.write(line)

    print("SRAM ACCESS COUNTS")
    print("IFM SRAM access: {}".format(num_IFM_acc))
    print("OFM SRAM access: {}".format(num_OFM_acc))
    print("Filt SRAM access (no kr): {}".format(num_filt_acc))
    print("Filt SRAM access (with kr): {}".format(num_filt_acc_kr))
    print("Total SRAM reads (with kr): {}".format(num_IFM_acc + num_filt_acc + num_OFM_acc))
    #print("Total SRAM writes (with kr): {}".format(num_IFM_acc + num_filt_acc_kr + num_OFM_acc))

    bw.close()
    maxbw.close()
    cycl.close()
    param_file.close()

#if __name__ == "__main__":
#    sweep_parameter_space_fast()    

