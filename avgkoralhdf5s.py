#!/usr/bin/env python
# Andrew Chael, March 2024
# phi-average, phi-slice, and t-average koral hdf5 files
# assumes axisymmetric metric
#
# Command line usage, see --help on each subcommand for the full option list:
#
#   avgkoralhdf5s.py avg   DIR         phi-average every dump, then time-average
#   avgkoralhdf5s.py slice DIR         phi-slice every dump at fixed phi index/indices
#   avgkoralhdf5s.py tavg  DIR|FILES   time-average 2D reduced files or full 3D dumps
#   avgkoralhdf5s.py all   DIR         avg (with time-average) followed by slice

import argparse
import glob
import os, re, sys
import numpy as np
from collections import OrderedDict
from metricKS import *
import h5py
from koralh5postproc import *
from koralopacities import *

# defaults, all overridable on the command line
# TODO MIGHT NOT HAVE ENOUGH MEMORY TO RUN IN PARALLEL FOR RADIATION
NPROC = 4           # processes to spread the files over; 1 = serial, 0 = all cores
METRIC = 'KS'       # coordinates of the output vector components
TMIN1 = 1.e4        # only phi-reduce dumps inside this time window
TMAX = 2.e4
TMIN2 = 1.5e4       # only time-average inside this time window
RERUN = True        # reprocess dumps whose output already exists
PHIIDX = 0          # phi index for 'slice'
INPATTERN = 'ipole*.h5'

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

def outname(infile, outpath, label):
    """Output name for a dump, keyed on its number: ipole5000 -> phiavg5000

    Takes the trailing digits of the input name, so it does not care how long
    the input prefix is, and falls back to the whole stem if there are none.
    """
    stem = os.path.splitext(os.path.basename(infile))[0]
    match = re.search(r'(\d+)$', stem)
    tag = match.group(1) if match else stem
    # keep a label that already ends in digits (phisli_ph009) from running into
    # the dump number; a plain label like phiavg is left alone
    sep = '_' if label and label[-1].isdigit() else ''
    return os.path.join(outpath, label + sep + tag + '.h5')


def gather_files(inputs, pattern):
    """Expand a list of directories, files or globs into a sorted file list.

    Files written by the time-average (*_tavg*) are always dropped, so that
    re-running never folds a previous average back into a new one.
    """
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files.extend(glob.glob(os.path.join(item, pattern)))
        else:
            hits = glob.glob(item)
            files.extend(hits if hits else [item])
    files = [f for f in files if '_tavg' not in os.path.basename(f)]
    return sorted(set(files))


def run_reduction(infiles, outpath, reducetype, label, phiidx, metric_avg,
                  tmin, tmax, rerun, kn, fields, exclude, nproc, verbose):
    """phi-reduce a list of dumps, in parallel if asked for"""
    outfiles = [outname(f, outpath, label) for f in infiles]
    args = [[infiles[i], outfiles[i], reducetype, phiidx, metric_avg,
             tmin, tmax, rerun, verbose, kn, fields, exclude]
            for i in range(len(infiles))]

    if nproc != 1 and len(args) > 0:
        try:
            import ehtim.parloop as parloop
        except ImportError:
            print('ehtim.parloop not available, running serially')
        else:
            ploop = parloop.Parloop(phireduce_hdf5)
            _ = ploop.run_loop(args, nproc)
            del ploop, _
            return outfiles

    for arg in args:
        phireduce_hdf5(*arg)

    return outfiles


def slice_labels(label, phiidxs):
    """One output prefix per phi index, left alone when there is only one"""
    if len(phiidxs) == 1:
        return {phiidxs[0]: label}
    return OrderedDict((idx, '%s_ph%03d' % (label, idx)) for idx in phiidxs)


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
        dump = read_koral_hdf53D(filein, metric=metric_avg, kn=kn, verbose=verbose)
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
            # a 3D dump needs its derived quantities computed before they can be
            # averaged; a 2D file already has them stored, and the extra
            # plotting quantities are not what we are averaging
            if is3d:
                src = read_koral_hdf53D(filein, metric=metric_avg, kn=kn, verbose=False)
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
            template = read_koral_hdf53D(templatefile, metric=metric_avg, kn=kn,
                                         verbose=False, compute_derived=False)
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


###############################################################################
# subcommands
###############################################################################

def do_avg(args):
    """phi-average every dump in the input directory, then time-average"""
    infiles = gather_files([args.inpath], args.pattern)
    if len(infiles) == 0:
        print('no input files matching', args.pattern, 'in', args.inpath)
        return

    print('phi-averaging %d files' % len(infiles))
    run_reduction(infiles, args.outpath, 'avg', args.label, PHIIDX, args.metric,
                  args.tmin, args.tmax, args.rerun, args.kn,
                  args.fields, args.exclude, args.nproc, args.verbose)

    if args.tavg:
        avgfiles = gather_files([args.outpath], args.label + '*.h5')
        tavg_hdf5s(avgfiles, os.path.join(args.outpath, args.label),
                   tmin=args.tavg_tmin, tmax=args.tavg_tmax,
                   metric_avg=args.metric, kn=args.kn,
                   fields=args.fields, exclude=args.exclude, verbose=args.verbose)
    return


def do_slice(args):
    """phi-slice every dump in the input directory, at each requested phi index"""
    infiles = gather_files([args.inpath], args.pattern)
    if len(infiles) == 0:
        print('no input files matching', args.pattern, 'in', args.inpath)
        return

    labels = slice_labels(args.label, args.phi_idx)
    for phiidx in args.phi_idx:
        print('phi-slicing %d files at phi index %d' % (len(infiles), phiidx))
        run_reduction(infiles, args.outpath, 'slice', labels[phiidx], phiidx, args.metric,
                      args.tmin, args.tmax, args.rerun, args.kn,
                      args.fields, args.exclude, args.nproc, args.verbose)
    return


def do_tavg(args):
    """time-average the given files, 2D reduced or full 3D"""
    pattern = args.pattern if args.pattern else 'phiavg*.h5'
    infiles = gather_files(args.inputs, pattern)
    if len(infiles) == 0:
        print('no input files matching', pattern, 'in', ' '.join(args.inputs))
        return

    # default the output prefix to the input prefix: phiavg5000 -> phiavg,
    # phisli_ph009_5000 -> phisli_ph009
    label = args.label
    if label is None:
        stem = os.path.splitext(os.path.basename(infiles[0]))[0]
        label = re.sub(r'[_-]?\d+$', '', stem) or stem

    outpath = args.outpath
    if outpath is None:
        outpath = os.path.dirname(os.path.abspath(infiles[0]))

    tavg_hdf5s(infiles, os.path.join(outpath, label),
               tmin=args.tmin, tmax=args.tmax,
               metric_avg=args.metric, kn=args.kn,
               fields=args.fields, exclude=args.exclude, verbose=args.verbose)
    return


def do_all(args):
    """avg, with its time-average, followed by slice"""
    do_avg(args)

    slice_args = argparse.Namespace(**vars(args))
    slice_args.label = args.slice_label
    do_slice(slice_args)
    return


###############################################################################
# command line
###############################################################################

def _csv(value):
    """--fields rho,bsq,sigma -> ['rho','bsq','sigma']"""
    return [v.strip() for v in value.split(',') if v.strip()]


def build_parser():
    fmt = argparse.ArgumentDefaultsHelpFormatter

    # option groups shared between subcommands
    io_p = argparse.ArgumentParser(add_help=False)
    io_p.add_argument('-o', '--outpath', default=None, metavar='DIR',
                      help='directory for the output files (default: the input directory)')
    io_p.add_argument('--pattern', default=INPATTERN, metavar='GLOB',
                      help='glob for the input dumps inside the input directory')

    sel_p = argparse.ArgumentParser(add_help=False)
    sel_p.add_argument('--tmin', type=float, default=TMIN1,
                       help='only process dumps with t >= TMIN')
    sel_p.add_argument('--tmax', type=float, default=TMAX,
                       help='only process dumps with t <= TMAX')

    phys_p = argparse.ArgumentParser(add_help=False)
    phys_p.add_argument('--metric', choices=['KS','BL'], default=METRIC,
                        help='coordinates of the output vector components')
    phys_p.add_argument('--kn', dest='kn', action='store_true', default=KLEINNISHINA,
                        help='include the Klein-Nishina correction in the opacities')
    # SUPPRESS on the negative half of each pair: the positive one already
    # supplies the default, and without it the help prints a misleading one
    phys_p.add_argument('--no-kn', dest='kn', action='store_false',
                        default=argparse.SUPPRESS,
                        help='leave the Klein-Nishina correction out')

    run_p = argparse.ArgumentParser(add_help=False)
    run_p.add_argument('-n', '--nproc', type=int, default=NPROC, metavar='N',
                       help='processes to spread the files over; 1 = serial, 0 = all cores')
    run_p.add_argument('--rerun', dest='rerun', action='store_true', default=RERUN,
                       help='reprocess dumps whose output file already exists')
    run_p.add_argument('--no-rerun', dest='rerun', action='store_false',
                       default=argparse.SUPPRESS,
                       help='skip dumps whose output file already exists')
    run_p.add_argument('-q', '--quiet', dest='verbose', action='store_false', default=True,
                       help='less chatter per file')

    fld_p = argparse.ArgumentParser(add_help=False)
    fld_p.add_argument('--fields', type=_csv, default=None, metavar='A,B,C',
                       help='only compute/average these quantities (default: all of them)')
    fld_p.add_argument('--exclude-fields', dest='exclude', type=_csv, default=None,
                       metavar='A,B,C', help='drop these quantities')

    tavg_p = argparse.ArgumentParser(add_help=False)
    tavg_p.add_argument('--tavg', dest='tavg', action='store_true', default=True,
                        help='time-average the phi-averages when finished')
    tavg_p.add_argument('--no-tavg', dest='tavg', action='store_false',
                        default=argparse.SUPPRESS,
                        help='stop after the phi-average')
    tavg_p.add_argument('--tavg-tmin', type=float, default=TMIN2,
                        help='start of the time-averaging window')
    tavg_p.add_argument('--tavg-tmax', type=float, default=TMAX,
                        help='end of the time-averaging window')

    sli_p = argparse.ArgumentParser(add_help=False)
    sli_p.add_argument('--phi-idx', type=int, action='append', default=None, metavar='I',
                       help='phi index to slice at; repeat it for several slices '
                            '(default: %d)' % PHIIDX)

    parser = argparse.ArgumentParser(
        prog='avgkoralhdf5s.py',
        description='phi-average, phi-slice and time-average KORAL ipole-format hdf5 dumps.')
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')

    p = sub.add_parser('avg', parents=[io_p, sel_p, phys_p, run_p, fld_p, tavg_p],
                       formatter_class=fmt,
                       help='phi-average every dump, then time-average the results')
    p.add_argument('inpath', help='directory holding the dumps')
    p.add_argument('--label', default='phiavg', help='prefix for the output files')
    p.set_defaults(func=do_avg)

    p = sub.add_parser('slice', parents=[io_p, sel_p, phys_p, run_p, fld_p, sli_p],
                       formatter_class=fmt,
                       help='phi-slice every dump at one or more fixed phi indices')
    p.add_argument('inpath', help='directory holding the dumps')
    p.add_argument('--label', default='phisli', help='prefix for the output files')
    p.set_defaults(func=do_slice)

    p = sub.add_parser('tavg', parents=[phys_p, fld_p], formatter_class=fmt,
                       help='time-average 2D reduced files or full 3D dumps')
    p.add_argument('inputs', nargs='+', metavar='PATH',
                   help='directory, files or globs to average')
    p.add_argument('--pattern', default=None, metavar='GLOB',
                   help='glob applied inside a directory argument (default: phiavg*.h5)')
    p.add_argument('-o', '--outpath', default=None, metavar='DIR',
                   help='directory for the output file (default: alongside the inputs)')
    p.add_argument('--label', default=None,
                   help='prefix for the output file (default: taken from the inputs)')
    p.add_argument('--tmin', type=float, default=TMIN2,
                   help='start of the time-averaging window')
    p.add_argument('--tmax', type=float, default=TMAX,
                   help='end of the time-averaging window')
    p.add_argument('-q', '--quiet', dest='verbose', action='store_false', default=True,
                   help='less chatter per file')
    p.set_defaults(func=do_tavg)

    p = sub.add_parser('all', parents=[io_p, sel_p, phys_p, run_p, fld_p, tavg_p, sli_p],
                       formatter_class=fmt,
                       help='avg, with its time-average, followed by slice')
    p.add_argument('inpath', help='directory holding the dumps')
    p.add_argument('--label', default='phiavg', help='prefix for the phi-averaged files')
    p.add_argument('--slice-label', default='phisli', help='prefix for the phi-sliced files')
    p.set_defaults(func=do_all)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # things the parser cannot express directly
    if getattr(args, 'phi_idx', None) is None and hasattr(args, 'phi_idx'):
        args.phi_idx = [PHIIDX]
    if getattr(args, 'inpath', None) is not None:
        args.inpath = os.path.join(args.inpath, '')
        if args.outpath is None:
            args.outpath = args.inpath
    if getattr(args, 'outpath', None) and not os.path.isdir(args.outpath):
        os.makedirs(args.outpath)

    args.func(args)
    return 0


if __name__=='__main__':
    sys.exit(main())
