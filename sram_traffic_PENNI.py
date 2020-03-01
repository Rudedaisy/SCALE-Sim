import math
from tqdm import tqdm

def sram_traffic(
        array_h = 7,
        array_w = 7,
        ifmap_h=7,
        ifmap_w=7,
        num_channels=4,
        num_bases=5,
        num_filt=128,
        coeff_ptrs=[],
        filt_base=1000000,
        ifmap_base=0,
        ofmap_base=3000000,
        wa_base=2000000,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv"
):

    read_cycles, util = gen_read_trace(
        array_h = array_h,
        array_w = array_w,
        ifmap_h = ifmap_h,
        ifmap_w = ifmap_w,
        num_channels = num_channels,
        num_bases = num_bases,
        num_filt = num_filt,
        coeff_ptrs = coeff_ptrs,
        filt_base = filt_base,
        ifmap_base = ifmap_base,
        ofmap_base = ofmap_base,
        wa_base = wa_base,
        sram_read_trace_file=sram_read_trace_file
    )
    
    write_cycles = gen_write_trace(
        array_h = array_h,
        array_w = array_w,
        ifmap_h = ifmap_h,
        ifmap_w = ifmap_w,
	num_channels = num_channels,
        num_bases = num_bases,
        num_filt = num_filt,
        coeff_ptrs = coeff_ptrs,
	filt_base = filt_base,
        ifmap_base = ifmap_base,
	ofmap_base = ofmap_base,
        wa_base = wa_base,
        sram_write_trace_file=sram_write_trace_file
        )

    cycles = max(read_cycles, write_cycles)
    str_cycles = str(int(cycles))
    return(str_cycles, util)

def gen_read_trace(
        array_h = 7,
        array_w = 7,
        ifmap_h=7,
        ifmap_w=7,
	num_channels=4,
        num_bases=5,
	num_filt=128,
        coeff_ptrs=[],
        filt_base=1000000,
        ifmap_base=0,
        ofmap_base=3000000,
        wa_base=2000000,
        sram_read_trace_file="sram_read.csv",
):

    cycle = 0
    util = 0
    IFM_addresses = []
    coeff_addresses = []
    IFM_done = False    
    row_offset = []

    # layer specific variables
    e2 = ifmap_h * ifmap_w
    bc = num_bases * num_channels

    # variables that change every loop
    pix_left = e2

    # initialize staggaring row-offset                                                                                               
    for r in range(array_h):
        row_offset.append(r * -1)

    # flatten coeff_ptrs list to allow iterations to cross out_channels
    flat_coeff_ptrs = [item for sublist in coeff_ptrs for item in sublist]
        
    # Open tracefile for writing
    outfile = open(sram_read_trace_file, 'w')
    
    # Adding progress bar
    tot  = math.ceil(e2 / (array_h * array_w)) * len(flat_coeff_ptrs)
    #print("Total = " + str(tot))
    pbar = tqdm(total=tot)
    

    ## prioritize parallelizing pixels across the scale_out (array width)
    while not IFM_done:

        ## initialize key loop variables
        # number of rows that can be filled in parallel this iteration
        full_rows = e2 // array_w
        partial_row = 0
        if full_rows < array_h:
            # number of columns filled in the last, partial row
            partial_row = e2 % array_w
        else:
            # max number of full rows is the array height
            full_rows = array_h

        # signify the marked columns
        parallel_columns = array_w
        if full_rows == 0:
            parallel_columns = partial_row

        # offset from finished pixels
        pix_offset = (e2 - pix_left)*bc
            
        ## loop to feed all coeffs and matching IFMs in a staggaring fashion
        for c_id, c_ptr in enumerate(flat_coeff_ptrs):
            IFM_addresses = []
            coeff_addresses = []

            # load all marked columns in parallel
            coeff_addresses.extend([c_id + filt_base] * parallel_columns)

            ## load all activated + marked rows in parallel
            # marked -> designated by full_ or partial_row
            # activated -> active row resulting from staggaring

            # this loop is for full_rows
            for r in range(full_rows):
                if r <= c_id:
                    for c in range(parallel_columns):
                        # note: out_layer[cid] = c_ptr, which has the form [base*num_channels + channel]
                        IFM_addresses.append(r*parallel_columns*bc + \
                                             c*bc + \
                                             pix_offset + \
                                             flat_coeff_ptrs[c_id + row_offset[r]] + \
                                             wa_base)

            # this loop is for partial_row
            if partial_row > 0:
                r = full_rows
                if r <= c_id:
                    for c in range(partial_row):
                        IFM_addresses.append(r*parallel_columns*bc + \
                                             c*bc + \
                                             pix_offset + \
                                             flat_coeff_ptrs[c_id + row_offset[r]] + \
                                             wa_base)

            ## update trace strings
            ifm_read = ""
            coeff_read = ""
            for c in range(array_w):
                if c < len(coeff_addresses):
                    coeff_read += str(int(coeff_addresses[c])) + ", "
                else:
                    coeff_read += ", "
                        
            for r in range(array_w * array_h):
                if r < len(IFM_addresses):
                    ifm_read += str(int(IFM_addresses[r])) + ", "
                else:
                    ifm_read += ", "

            ## write to trace file
            entry = str(cycle) + ", " + ifm_read + coeff_read + "\n"
            outfile.write(entry)
                            
            # update loop variables
            cycle += 1
            pbar.update(1)

        pix_left -= ((full_rows * array_w) + partial_row)
        
        if pix_left == 0:
            ## empty-out the remaining IFM window feeds
            for flush_it in range(array_h):
                IFM_addresses = []
                coeff_addresses = []
            
                # full_rows
                for r in range(full_rows):
                    if r > flush_it:
                        for c in range(parallel_columns):
                            IFM_addresses.append(r*parallel_columns*bc + \
                                                 c*bc + \
                                                 pix_offset + \
                                                 flat_coeff_ptrs[len(flat_coeff_ptrs) + flush_it + row_offset[r]] + \
                                                 wa_base)
                # partial_row
                if partial_row > 0:
                    r = full_rows
                    if r > flush_it:
                        for c in range(partial_row):
                            IFM_addresses.append(r*parallel_columns*bc + \
                                                 c*bc + \
                                                 pix_offset + \
                                                 flat_coeff_ptrs[len(flat_coeff_ptrs) + flush_it + row_offset[r]] + \
	                                         wa_base)

                ## update trace strings
                ifm_read = ""
                coeff_read = ""
                for c in range(array_w):
                    # coefficients are no longer being fed in
                    coeff_read += ", "
                for r in range((flush_it * -1), (array_w * array_h) - flush_it):
                    if r >= 0 and r < len(IFM_addresses):
                        ifm_read += str(int(IFM_addresses[r])) + ", "
                    else:
                        ifm_read += ", "
                            
                ## write to trace file
                entry = str(cycle) + ", " + ifm_read + coeff_read + "\n"
                outfile.write(entry)

                # update local variable
                cycle += 1
                
            IFM_done = True
        
    pbar.close()
    outfile.close()

    util_perc = 0.0 ###### TODO: generate real number for util...
    return cycle, util_perc

def gen_write_trace(
        array_h = 7,
        array_w = 7,
        ifmap_h=7,
        ifmap_w=7,
	num_channels=4,
        num_bases=5,
        num_filt=128,
        coeff_ptrs=[],
	filt_base=1000000,
        ifmap_base=0,
        ofmap_base=3000000,
        wa_base=2000000,
	sram_write_trace_file="sram_write.csv",
):
    cycle = 0
    OFM_addresses = []
    IFM_done = False
    row_offset = []
    
    # layer specific variables
    e2 = ifmap_h * ifmap_w
    bc = num_bases * num_channels

    # variables that change every loop
    pix_left = e2

    # initialize staggaring row-offset                                                                                               
    for r in range(array_h):
        row_offset.append(r * -1)
    
    # Open the file for writing
    outfile = open(sram_write_trace_file, 'w')

    """
    while not IFM_done:
        full_rows = e2 // array_w
        partial_row = 0
        if full_rows < array_h:
            # number of columns filled in the last, partial row
            partial_row = e2 % array_w
        else:
            # max number of full rows is the array height
            full_rows = array_h

        # signify the marked columns
        parallel_columns = array_w
        if full_rows == 0:
            parallel_columns = partial_row

        # offset from the finished pixels
        pix_offset = e2 - pix_left
    """
    OFM_address = [1,2,3,4,5,6]
        

    # Write to trace file
    entry = str(cycle) + ", " + str(int(OFM_address[0] + ofmap_base)) + "\n"
    outfile.write(entry)
    
    # We assume 1 cycle to write result
    cycle += 1

    outfile.close()
    return cycle
