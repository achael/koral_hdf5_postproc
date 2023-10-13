# make a time-average and phi-average koral hdf5 file
# MHD only
# assumes axisymmetric metric

import glob
import os
import numpy as np
from metricKS import *
from read_phiavg_BLallquants import read_phiavg_hdf5
import h5py

in_path = './ap5ff_0517/analysis/'
out_path = './ap5ff_0517/analysis/'
label = 'ff'

#TMIN = 50000.
#TMAX = 100000.

TMIN = 5000.
TMAX = 10000.

METRIC_OUT = 'BL'

def main():
     

    infiles = np.sort(glob.glob(in_path + label + '_ipole*phiavg.h5'))
    print(infiles)
    
    navg = 0
    for filein in infiles:
        fin = h5py.File(filein,'r')
        time = fin['t'][()]
        #print(time)
        fin.close()
        if time<TMIN or time>TMAX:
            continue
        else:
            datdict = read_phiavg_hdf5(filein, metric=METRIC_OUT)
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

        outfile = out_path + label + '_ipole_tavg%.0f-%.0f.h5'%(TMIN,TMAX)
        
        save_tphiavg_hdf5(outfile, infiles[0], avgdict)
    else:
        print('no files in average')
             
    return

def save_tphiavg_hdf5(fileout, templatefile, avgdict, metric=METRIC_OUT, verbose=True):

    if verbose: print('saving time-averaged hdf5 ', fileout, '....')
    
    # load template data     
    fin = h5py.File(templatefile,'r')

    # get info from the header
    spin = fin['header']['bhspin'][()].astype('f')
    horiz = 1 + np.sqrt(1-spin**2)
    metric_run = fin['header']['metric_run'][()]
    metric_out = fin['header']['metric_out'][()]
    if not isinstance(metric_run, str): metric_run = metric_run.decode('utf-8')
    if not isinstance(metric_out, str): metric_out = metric_out.decode('utf-8')
    if metric_out != metric:
        raise Exception("output metric %s must be %s!"%(metric_out,metric))

    NX = fin['header']['n1'][()]
    NY = fin['header']['n2'][()]
    NZ = fin['header']['n3'][()]
    gamma_adiab = fin['header']['gam'][()]

    # get coordinates in OUTCOORDS (should always be KS)
    r = fin['grid_out']['r']
    th = fin['grid_out']['th']
    ph = fin['grid_out']['ph']

    ############################################################################################
    # make a new hdf5 file
    fout = h5py.File(fileout,'w')
    try:
        # Time
        fout.create_dataset('t',data='tavg%.0f-%.0f'%(TMIN,TMAX))
        
        # Header
        grp = fout.create_group('header')    
        grp.create_dataset('bhspin',data=spin)
        grp.create_dataset('file_number',data='tavg%.0f-%.0f'%(TMIN,TMAX))
        grp.create_dataset('gam',data=gamma_adiab)    
        grp.create_dataset('has_electrons',data=0) # MHD only for now
        grp.create_dataset('has_radiation',data=0) # MHD only for now    
        grp.create_dataset('metric_out',data=metric_out) 
        grp.create_dataset('metric_run',data=fin['header']['metric_run'][()])
        grp.create_dataset('n1',data=NX) 
        grp.create_dataset('n2',data=NY) 
        grp.create_dataset('n3',data=1)  #phi-averaged
        grp.create_dataset('ndim',data=2)  #phi-averaged
        grp.create_dataset('problem_number',data=fin['header']['problem_number'][()])     
        grp.create_dataset('version',data=fin['header']['version'][()])
        
        grp.copy(fin['header']['units'], grp)
        grp.copy(fin['header']['geom'], grp)
        
        # Grid
        grp = fout.create_group('grid_out')
        grp.create_dataset('r',  data=r)
        grp.create_dataset('th', data=th)
        grp.create_dataset('ph', data=ph)
        
        # Primitive Quantities
        grp = fout.create_group('quants')
        grp.create_dataset('rho',  data=avgdict['rho'])
        grp.create_dataset('uint', data=avgdict['uint'])
        
        grp.create_dataset('B1', data=avgdict['B1'])
        grp.create_dataset('B2', data=avgdict['B2'])
        grp.create_dataset('B3', data=avgdict['B3'])
        
        grp.create_dataset('U1', data=avgdict['U1'])
        grp.create_dataset('U2', data=avgdict['U2'])
        grp.create_dataset('U3', data=avgdict['U3'])
        
        # Derived Quantities
        grp.create_dataset('lorentz', data=avgdict['lorentz'])
        grp.create_dataset('bsq',     data=avgdict['bsq'])
        grp.create_dataset('sigma',   data=avgdict['sigma']) 
        grp.create_dataset('beta',    data=avgdict['beta']) 
        grp.create_dataset('Tgas',    data=avgdict['Tgas'])
        grp.create_dataset('betainv', data=avgdict['betainv'])
        
        grp.create_dataset('b0', data=avgdict['b0'])
        grp.create_dataset('b1', data=avgdict['b1'])
        grp.create_dataset('b2', data=avgdict['b2'])
        grp.create_dataset('b3', data=avgdict['b3'])
        
        grp.create_dataset('u0', data=avgdict['u0'])
        grp.create_dataset('u1', data=avgdict['u1'])
        grp.create_dataset('u2', data=avgdict['u2'])
        grp.create_dataset('u3', data=avgdict['u3'])

        grp.create_dataset('sF12',  data= avgdict['sF12'])    
        grp.create_dataset('sF13',  data= avgdict['sF13'])    
        grp.create_dataset('sF23',  data= avgdict['sF23'])    
        
        grp.create_dataset('rhou1', data= avgdict['rhou1'])    
        
        grp.create_dataset('T00_mag',  data= avgdict['T00_mag'])    
        grp.create_dataset('T01_mag',  data= avgdict['T01_mag'])    
        grp.create_dataset('T02_mag',  data= avgdict['T02_mag'])    
        grp.create_dataset('T03_mag',  data= avgdict['T03_mag'])    
        grp.create_dataset('T11_mag',  data= avgdict['T11_mag'])    
        grp.create_dataset('T12_mag',  data= avgdict['T12_mag'])    
        grp.create_dataset('T13_mag',  data= avgdict['T13_mag'])    
        grp.create_dataset('T22_mag',  data= avgdict['T22_mag'])    
        grp.create_dataset('T23_mag',  data= avgdict['T23_mag'])    
        grp.create_dataset('T33_mag',  data= avgdict['T33_mag'])    
                                                        
        
        grp.create_dataset('T00_hd',  data= avgdict['T00_hd'])    
        grp.create_dataset('T01_hd',  data= avgdict['T01_hd'])    
        grp.create_dataset('T02_hd',  data= avgdict['T02_hd'])    
        grp.create_dataset('T03_hd',  data= avgdict['T03_hd'])    
        grp.create_dataset('T11_hd',  data= avgdict['T11_hd'])    
        grp.create_dataset('T12_hd',  data= avgdict['T12_hd'])    
        grp.create_dataset('T13_hd',  data= avgdict['T13_hd'])    
        grp.create_dataset('T22_hd',  data= avgdict['T22_hd'])    
        grp.create_dataset('T23_hd',  data= avgdict['T23_hd'])    
        grp.create_dataset('T33_hd',  data= avgdict['T33_hd'])    
    except:
        print("Error writing averaged hdf5 file ", fileout, "!")   
    fin.close()
    fout.close()
    
    return

if __name__=='__main__':
    main()
