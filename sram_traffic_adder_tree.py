import math
from tqdm import tqdm

def sram_traffic(
        ifmap_h=7,
        ifmap_w=7,
        add_tree_leaves=512,
        num_channels=4,
        num_bases=5,
        num_filt=128,
        coeff_ptrs=[],
        filt_base=1000000,
        ifmap_base=0,
        ofmap_base=2000000,
        coeff_ptr_base=3000000,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv"
):
    # each cycle is dedicated to loading coeffs and IFMs for ONE output channel (of one pixel)
    adder_depth = math.log2(add_tree_leaves) + 1
    assert adder_depth.is_integer()

    read_cycles, util = gen_read_trace(
        ifmap_h = ifmap_h,
        ifmap_w = ifmap_w,
        add_tree_leaves = add_tree_leaves,
        adder_depth = adder_depth,
        num_channels = num_channels,
        num_bases = num_bases,
        num_filt = num_filt,
        coeff_ptrs = coeff_ptrs,
        filt_base = filt_base,
        ifmap_base = ifmap_base,
        ofmap_base = ofmap_base,
        coeff_ptr_base = coeff_ptr_base,
        sram_read_trace_file=sram_read_trace_file
    )

    write_cycles = gen_write_trace(
        ifmap_h = ifmap_h,
        ifmap_w = ifmap_w,
	add_tree_leaves = add_tree_leaves,
	adder_depth = adder_depth,
	num_channels = num_channels,
        num_bases = num_bases,
        num_filt = num_filt,
        coeff_ptrs = coeff_ptrs,
	filt_base = filt_base,
        ifmap_base = ifmap_base,
	ofmap_base = ofmap_base,
        coeff_ptr_base = coeff_ptr_base,
        sram_write_trace_file=sram_write_trace_file
        )

    cycles = max(read_cycles, write_cycles)
    str_cycles = str(int(cycles))
    return(str_cycles, util)

def gen_read_trace(
        ifmap_h=7,
        ifmap_w=7,
        add_tree_leaves=512,
        adder_depth=9,
	num_channels=4,
        num_bases=5,
	num_filt=128,
        coeff_ptrs=[],
        filt_base=1000000,
        ifmap_base=0,
        ofmap_base=2000000,
        coeff_ptr_base=3000000,
        sram_read_trace_file="sram_read.csv",
):

    cycle = 0
    util = 0
    IFM_addresses = []
    coeff_addresses = []
    # track the current coefficient ID; this acts as our "weight" address
    coeff_idx = 0
    
    # Open tracefile for writing
    outfile = open(sram_read_trace_file, 'w')

    # Adding progress bar
    tot  = num_filt * ifmap_h * ifmap_w
    #print("Total = " + str(tot))
    pbar = tqdm(total=tot)
    
    # we assume one adder tree structure for now...
    # TODO: allow tree to be smaller than number of coeffs for an output (i.e. include multiple passes)
    for h in range(ifmap_h):
        for w in range(ifmap_w):
            coeff_idx = 0
            for out_channel in coeff_ptrs:
                IFM_addresses = []
                coeff_addresses = []
                for c in out_channel:
                    b = c // num_channels
                    in_channel_idx = c % num_channels
                    IFM_address = h*ifmap_w*num_bases*num_channels + \
                                  w*num_bases*num_channels + \
                                  b*num_channels + \
                                  in_channel_idx
                    IFM_addresses.append(IFM_address + ifmap_base)
                    coeff_addresses.append(coeff_idx + filt_base)
                    coeff_idx += 1

                ifm_read = ""
                coeff_read = ""
                # TODO: Remove this safety net once variable sized tree works!
                assert len(IFM_addresses) <= add_tree_leaves
                util += len(IFM_addresses)
                # Update trace strings
                for leaf_idx in range(add_tree_leaves):
                    if leaf_idx < len(IFM_addresses):
                        ifm_read += str(int(IFM_addresses[leaf_idx])) + ", "
                        coeff_read += str(int(coeff_addresses[leaf_idx])) + ", "
                    else:
                        ifm_read += ", "
                        coeff_read += ", "

                # Write to trace file
                entry = str(cycle) + ", " + ifm_read + coeff_read + "\n"
                outfile.write(entry)
                
                # we assume loading all values to the tree takes 1 cycle
                cycle += 1
                # Update the progress bar
                pbar.update(1)

    pbar.close()
    outfile.close()

    util /= (num_filt * ifmap_h * ifmap_w * add_tree_leaves)
    util_perc = util * 100

    return cycle, util_perc

def gen_write_trace(
        ifmap_h=7,
        ifmap_w=7,
	add_tree_leaves=512,
        adder_depth=9,
	num_channels=4,
        num_bases=5,
        num_filt=128,
        coeff_ptrs=[],
	filt_base=1000000,
        ifmap_base=0,
        ofmap_base=2000000,
        coeff_ptr_base=3000000,
	sram_write_trace_file="sram_write.csv",
):
    # This is the cycle when the first accumulation result becomes available
    cycle = adder_depth - 1

    # Open the file for writing
    outfile = open(sram_write_trace_file, 'w')

    for h in range(ifmap_h):
        for w in range(ifmap_w):
            coeff_idx = 0
            for out_channel_idx in range(num_filt):
                OFM_address = out_channel_idx*ifmap_h*ifmap_w + \
                              h*ifmap_w + \
                              w

                # Write to trace file
                entry = str(cycle) + ", " + str(int(OFM_address + ofmap_base)) + "\n"
                outfile.write(entry)

                # We assume 1 cycle to write result
                cycle += 1

    outfile.close()
    return cycle
