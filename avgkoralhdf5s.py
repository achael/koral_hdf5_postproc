# Andrew Chael, March 2024
# phi-average, phi-slice, and t-average koral hdf5 files
# MHD only!!!!
# assumes axisymmetric metric

import glob
import os, sys
import numpy as np
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

    return


def tavg_hdf5s(infilelist, outfilebase, tmin=TMIN2, tmax=TMAX):
    """time-average phi-averaged hdf5 files"""
    navg = 0
    tminfile=1.e100
    tmaxfile=0
    for filein in infilelist:
        fin = h5py.File(filein,'r')
        time = fin['t'][()]
        fin.close()
        if time<tmin or time>tmax:
            continue
        else:
            if time<tminfile: tminfile=time
            if time>tmaxfile: tmaxfile=time
            print(time)
            koraldata = read_koral_hdf52D(filein, verbose=False)
            datdict = koraldata.data
            
            if navg==0:
                avgdict = datdict.copy()
            else:
                for field in avgdict.keys():
                    avgdict[field] += datdict[field]
            navg += 1
             
    if navg>0:
        print('averaging ',navg,'files')
        for field in avgdict.keys():
            avgdict[field] /= float(navg)

        outfile = os.path.splitext(outfilebase)[0] + '_tavg%.0f-%.0f.h5'%(tminfile,tmaxfile)
        
        # save the file
        print('saving time-averaged hdf5 ', outfile, '....')
        
        # load template data     
        fin = h5py.File(infilelist[0],'r')
            
        # open output file
        fout = h5py.File(outfile,'w')
        
        # Time
        fout.create_dataset('t',data='tavg%.0f-%.0f'%(tminfile,tmaxfile))
        
        # Copy header 
        fout.copy(fin['header'],fout)
    
        # Copy grid
        fout.copy(fin['grid_out'],fout)
        
        # save quants
        grp = fout.create_group('quants')
        for key in avgdict.keys():
            grp[key] = avgdict[key]        

        # close
        fin.close()
        fout.close()
    
    
    else:
        print('no files in average')
             
    return
    
if __name__=='__main__':
    inpath = os.path.join(sys.argv[1],'')
    outpath = inpath
    
    main(inpath, outpath, reducetype='avg')
    main(inpath, outpath, reducetype='slice', phiidx=PHIIDX)
    main(inpath, outpath, reducetype='tavg')
