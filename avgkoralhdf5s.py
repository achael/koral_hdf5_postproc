# Andrew Chael, March 2024
# phi-average, phi-slice, and t-average koral hdf5 files
# MHD only!!!!
# assumes axisymmetric metric

import glob
import os, sys
import numpy as np
from collections import OrderedDict
from metricKS import *
import h5py
import ehtim.parloop as parloop
from koralh5postproc import *
from koralopacities import *

# TODO MIGHT NOT HAVE ENOUGH MEMORY TO RUN IN PARALLEL FOR RADIATION
NPROC = 4

# paths
LIBPATH = './' # use sys.argv[1] in main()
OUTPATH = './' # same as inpath in main()

METRIC ='KS'
TMIN1= 1.e4
TMIN2=1.5e4
TMAX = 2.e4
RERUN = True
PHIIDX = 0


KLEINNISHINA=False # we had KN turned off in KORAL runs for some reason...

# pure hydrogen is hard coded
MU_I = 1
MU_E = 1
MU_GAS = 0.5

# constants
MSUN_CGS = 1.989e33
MP_CGS = 1.67262158e-24
ME_CGS = 9.1094e-28
MRATIO = MP_CGS / ME_CGS
C_CGS = 2.9979246e10
KBOLTZ_CGS = 1.3806488e-16
SIGMA_RAD_CGS = 5.670367e-5
A_RAD_CGS = 4*SIGMA_RAD_CGS/C_CGS
TPFAC = KBOLTZ_CGS/(MP_CGS*C_CGS*C_CGS) # kelvin to dimensionless units for Tp
TEFAC = KBOLTZ_CGS/(ME_CGS*C_CGS*C_CGS) # kelvin to dimensionless units for Te

def main(inpath=LIBPATH, outpath=OUTPATH, reducetype='avg', phiidx=PHIIDX, metric_avg=METRIC, tmin=TMIN1, tmax=TMAX, rerun=RERUN):
        
    if metric_avg not in ['KS','BL']:
        raise Exception("metric_avg must be 'KS' or 'BL'")

    if reducetype=='tavg':
        avgfiles = np.sort(glob.glob(os.path.join(outpath,'phiavg*.h5')))
        tavg_hdf5s(avgfiles, os.path.join(outpath, 'phiavg'), tmin=TMIN2, tmax=TMAX)

    else:
        if reducetype=='avg':
            label = 'phiavg'
        elif reducetype=='slice':
            label = 'phisli'
        else:
            raise Exception("reducetype must be 'avg' or 'slice")
                
        infiles = np.sort(glob.glob(os.path.join(inpath, 'ipole*.h5')))
        outfiles = [os.path.join(outpath, label + os.path.splitext(os.path.basename(file))[0][5:] + '.h5') for file in infiles]
    
        if NPROC>0:                
            args = [[infiles[i], outfiles[i], reducetype, phiidx, metric_avg, tmin, tmax, rerun, False] for i in range(len(infiles))]
            ploop = parloop.Parloop(phireduce_hdf5)
            _ = ploop.run_loop(args, NPROC)

            del args, ploop, _
        else:
            for i in range(len(infiles)):
                phireduce_hdf5(infiles[i],outfiles[i], reducetype, phiidx, metric_avg, tmin, tmax, rerun, False)
    
        # time average
        if reducetype=='avg':
            avgfiles = np.sort(glob.glob(os.path.join(outpath,label+'*.h5')))
            tavg_hdf5s(avgfiles, os.path.join(outpath, label), tmin=TMIN2, tmax=TMAX)
         
    return

def phireduce(data, reducetype='avg', phiidx=PHIIDX):
    """Either average or slice a 3D data set"""
    if reducetype=='avg':
        datareduced = np.nanmean(data,  axis=2)
    elif reducetype=='slice':
        datareduced = data[:,:,phiidx]
    else:
        raise Exception("reducetype must be 'avg' or 'slice'")
    return datareduced       
                   
def peek_time(filein):
    """Read /t out of a koral hdf5 file without loading anything else"""
    try:
        with h5py.File(filein, 'r') as fin:
            return float(fin['t'][()])
    except Exception:
        return None


def phireduce_hdf5(filein, fileout, reducetype='avg', phiidx=PHIIDX, metric_avg=METRIC,
                   tmin=TMIN1, tmax=TMAX, rerun=True, verbose=True,
                   kn=KLEINNISHINA, fields=None, exclude=None):
    """Either phi-average or phi-slice a koral hdf5 file"""
    if reducetype not in ['avg','slice']:
        raise Exception("reducetype must be 'avg' or 'slice'")
    if metric_avg not in ['KS','BL']:
        raise Exception("metric_avg must be 'KS' or 'BL'")
    if (not rerun) and os.path.exists(fileout):
        if verbose: print('skipping existing ', fileout)
        return

    # time - do not process if out of range
    time = peek_time(filein)
    if time is None:
        print("Error reading time from h5 file ", filein, "!")
        return
    if time < tmin or time > tmax:
        return

    if verbose: print('reducing hdf5 ', filein, '....')

    # load the dump and compute the derived quantities
    try:
        dump = simdata3D(filein, metric=metric_avg, kn=kn, verbose=verbose)
        dump.set_derived_quantities()
    except Exception as e:
        print("Error reading h5 file ", filein, ": ", e)
        return

    if reducetype == 'slice' and not (0 <= phiidx < dump.n3):
        print("phi index ", phiidx, " out of range [0,", dump.n3, ") for ", filein, "!")
        return

    # reduce function
    def pr(data):
        return phireduce(data, reducetype=reducetype, phiidx=phiidx)

    ###########################################################################################
    # make the output hdf5 file
    try:
        with h5py.File(fileout, 'w') as fout:
            fout.create_dataset('t', data=time)

            # Header - phi-reduced, so the phi axis is gone
            dump.write_header(fout, n3=1, ndim=2)

            # Grid
            grp = fout.create_group('grid_out')
            grp.create_dataset('r',  data=pr(dump.r))
            grp.create_dataset('th', data=pr(dump.th))
            if reducetype == 'slice':
                phidata = dump.ph[:,:,phiidx]
            else:
                phidata = np.zeros((dump.n1, dump.n2))
            grp.create_dataset('ph', data=phidata)

            # reduced and derived quantities, one at a time
            grp = fout.create_group('quants')
            for name in dump.field_names(fields, exclude):
                grp.create_dataset(name, data=pr(dump.field(name)))
    except Exception as e:
        print("Error writing reduced hdf5 file ", fileout, ": ", e)
        if os.path.exists(fileout):
            os.remove(fileout)
        return
    finally:
        # a derived dump is several GB, so hand it back before the next file
        dump.close()

    return


def is_3d_hdf5(filein):
    """True for a full 3D dump, False for a 2D phi-reduced file"""
    with h5py.File(filein, 'r') as fin:
        return len(fin['grid_out']['r'].shape) == 3


def tavg_hdf5s(infilelist, outfilebase, tmin=TMIN2, tmax=TMAX,
               metric_avg=METRIC, kn=KLEINNISHINA, fields=None, exclude=None,
               verbose=True):
    """Time-average a list of koral hdf5 files.

    Takes either 2D phi-reduced files (phiavg*/phisli*), whose stored quantities
    are averaged directly, or full 3D dumps, whose derived quantities are
    recomputed per dump and averaged on the 3D grid.  Which of the two is
    decided by looking at the first file.

    Files are opened one at a time and each quantity is accumulated as it is
    read, so peak memory is one running sum plus one dump.  That matters for 3D
    input, where a single dump is already ~0.7 GB; use fields/exclude to cut the
    running sum down further.

    Returns the name of the file written, or None.
    """
    infilelist = list(infilelist)
    if len(infilelist) == 0:
        print('no files to average')
        return None

    try:
        is3d = is_3d_hdf5(infilelist[0])
    except Exception as e:
        print("Error reading h5 file ", infilelist[0], ": ", e)
        return None

    accum = None
    names = None
    templatefile = None
    navg = 0
    tminfile = 1.e100
    tmaxfile = 0
    tsum = 0.

    for filein in infilelist:
        time = peek_time(filein)
        if time is None:
            print("Error reading time from h5 file ", filein, ", skipping!")
            continue
        if time < tmin or time > tmax:
            continue

        try:
            if is3d:
                src = simdata3D(filein, metric=metric_avg, kn=kn, verbose=False)
                src.set_derived_quantities()
            else:
                src = read_koral_hdf52D(filein, verbose=False, compute_derived=False)
            srcnames = src.field_names(fields, exclude)
        except Exception as e:
            print("Error reading h5 file ", filein, ", skipping: ", e)
            continue

        if accum is None:
            names = srcnames
            templatefile = filein
            accum = OrderedDict((k, None) for k in names)
            if verbose:
                nbytes = len(names)*np.prod(src.shape)*8
                print('averaging %d quantities, %.2f GB running sum'
                      % (len(names), nbytes/1.e9))
        elif srcnames != names:
            print("quantities in ", filein, " do not match ", templatefile,
                  ", skipping!")
            continue

        print(time)
        for name in names:
            arr = np.asarray(src.field(name), dtype=np.float64)
            if accum[name] is None:
                accum[name] = arr.copy()
            else:
                accum[name] += arr

        if time < tminfile: tminfile = time
        if time > tmaxfile: tmaxfile = time
        tsum += time
        navg += 1

        # let the dump go before opening the next one
        src.close()
        del src

    if navg == 0:
        print('no files in average')
        return None

    print('averaging ', navg, 'files')
    for name in names:
        accum[name] /= float(navg)

    outfile = os.path.splitext(outfilebase)[0] + '_tavg%.0f-%.0f.h5'%(tminfile,tmaxfile)

    # save the file
    print('saving time-averaged hdf5 ', outfile, '....')

    # reopen the first file in the average as the header/grid template.  Doing it
    # now rather than holding the object through the loop keeps its derived
    # quantities -- several GB for a 3D dump -- out of the running peak.
    try:
        if is3d:
            template = simdata3D(templatefile, metric=metric_avg, kn=kn, verbose=False)
        else:
            template = read_koral_hdf52D(templatefile, verbose=False, compute_derived=False)
    except Exception as e:
        print("Error reopening template file ", templatefile, ": ", e)
        return None

    try:
        with h5py.File(outfile, 'w') as fout:
            # Time - the mean, with the window and the file count alongside it
            fout.create_dataset('t', data=tsum/float(navg))
            fout.create_dataset('t_min', data=tminfile)
            fout.create_dataset('t_max', data=tmaxfile)
            fout.create_dataset('n_avg', data=navg)

            # Copy header and grid from the first file in the average
            template.write_tavg_header_and_grid(fout)

            # save quants
            grp = fout.create_group('quants')
            for name in names:
                grp.create_dataset(name, data=accum[name])
    except Exception as e:
        print("Error writing time-averaged hdf5 file ", outfile, ": ", e)
        if os.path.exists(outfile):
            os.remove(outfile)
        return None

    return outfile
    
if __name__=='__main__':
    inpath = os.path.join(sys.argv[1],'')
    outpath = inpath
    
    main(inpath, outpath, reducetype='avg')
    main(inpath, outpath, reducetype='slice', phiidx=PHIIDX)
    main(inpath, outpath, reducetype='tavg')
