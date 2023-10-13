# make a time-average and phi-average koral hdf5 file
# MHD only!!!!
# assumes axisymmetric metric

import glob
import os, os.path
import numpy as np
from metricKS import *
import h5py
import ehtim.parloop as parloop

NPROC = 0 # use maximum available

KS2BL = True
LIBPATH = './'
OUTPATH = './'
LABEL = 'label'
TMIN = 0
TMAX = 1.e100
RERUN = True

def main(inpath=LIBPATH, outpath=OUTPATH, label='label', ks2bl=KS2BL, tmin=TMIN, tmax=TMAX, rerun=RERUN):
        
    fun = phiavg_hdf5
    
    infiles = np.sort(glob.glob(inpath + 'ipole*.h5'))
    outfiles = [out_path  + label + '_' + os.path.splitext(os.path.basename(file))[0] + '_phiavg.h5'
                for file in infiles]
    
    if NPROC>0:                
        args = [[infiles[i], outfiles[i], ks2bl, tmin, tmax, rerun, False] for i in range(len(infiles))]
        ploop = parloop.Parloop(fun)
        _ = ploop.run_loop(args, NPROC)

        del args, ploop, _
    else:
        for i in range(len(infiles)):
            fun(infiles[i],outfiles[i], ks2bl, tmin, tmax, rerun, False)
            
    return

            
def phiavg_hdf5(filein, fileout, ks2bl=KS2BL,tmin=TMIN,tmax=TMAX, verbose=True):
    if verbose: print('averaging hdf5 ', filein, '....')
    
    if (not RERUN) and os.path.exists(fileout):
        return
         
    # load data     
    fin = h5py.File(filein,'r')

    # time
    time = fin['t'][()]

    
    if time<tmin or time>tmax:
        fin.close()
        return
                 
    # get info from the header
    spin = fin['header']['bhspin'][()].astype('f')
    horiz = 1 + np.sqrt(1-spin**2)
    metric_run = fin['header']['metric_run'][()]
    metric_out = fin['header']['metric_out'][()]
    if not isinstance(metric_run, str): metric_run = metric_run.decode('utf-8')
    if not isinstance(metric_out, str): metric_out = metric_out.decode('utf-8') 
    if metric_out != 'KS':
        raise Exception("metric_out must be KS!")

    NX = fin['header']['n1'][()]
    NY = fin['header']['n2'][()]
    NZ = fin['header']['n3'][()]
    gamma_adiab = fin['header']['gam'][()]

    # get coordinates in OUTCOORDS (should be KS)
    r = fin['grid_out']['r'][:,:,:]
    th = fin['grid_out']['th'][:,:,:]
    ph = fin['grid_out']['ph'][:,:,:]
    
    # get simulation primitive variables
    rho = fin['quants']['rho'][:,:,:]
    uint = fin['quants']['uint'][:,:,:]

    u1_velr = fin['quants']['U1'][:,:,:] 
    u2_velr = fin['quants']['U2'][:,:,:]
    u3_velr = fin['quants']['U3'][:,:,:]

    B1 = fin['quants']['B1'][:,:,:]
    B2 = fin['quants']['B2'][:,:,:]
    B3 = fin['quants']['B3'][:,:,:]
    
    # get 4-velocity and lorentz factor in KS
    (u0,u1,u2,u3) = conv_vel_KS(u1_velr, u2_velr, u3_velr, spin, r, th)
    (u0_l,u1_l,u2_l,u3_l) = lowerKS(u0, u1, u2, u3, spin, r, th)
    #check = u0*u0_l + u1*u1_l + u2*u2_l + u3*u3_l   

    # get magnetic field 4-vector and b^2 in KS
    b0 = u1_l*B1 + u2_l*B2 + u3_l*B3
    b1 = (B1 + b0*u1)/u0
    b2 = (B2 + b0*u2)/u0
    b3 = (B3 + b0*u3)/u0

    (b0_l,b1_l,b2_l,b3_l) = lowerKS(b0, b1, b2, b3, spin, r, th)
    bsq = b0*b0_l + b1*b1_l + b2*b2_l + b3*b3_l
    
    # plasma parameters
    pgas = (gamma_adiab-1)*uint
    w = (rho + pgas + uint)
    Tgas = pgas / rho # dimensionless temperature
    beta = pgas / (0.5*bsq)
    betainv = (0.5*bsq) / pgas
    sigma = bsq / rho
    
    # convert to BL
    if ks2bl:
        metric_out = "BL"
        gcon = gconBL(spin,r,th)
        
        (u0,u1,u2,u3) = trans_cov_ks2bl(u0, u1, u2, u3, spin, r, th)
        (u0_l, u1_l, u2_l, u3_l) = lowerBL(u0,u1,u2,u3,spin,r,th)
        (b0,b1,b2,b3) = trans_cov_ks2bl(b0, b1, b2, b3, spin, r, th)
        (b0_l, b1_l, b2_l, b3_l) = lowerBL(b0,b1,b2,b3,spin,r,th)
        
        u1_velr, u2_velr, u3_velr = invconv_vel_BL(u0, u1, u2, u3, spin, r, th)
        lorentz = lorentz_BL(u0, spin, r, th)
    else:     
        metric_out = "KS"
        gcon = gconKS(spin,r,th)
        lorentz = lorentz_KS(u0, spin, r, th)
        
    # (re) derive B-field 3 vector
    B1 = b1*u0 - b0*u1
    B2 = b2*u0 - b0*u2
    B3 = b3*u0 - b0*u3  

    # star F (contravariant)
    #sF01 = -B1 # not necessary to save twice
    #sF02 = -B2
    #sF03 = -B3
    sF12 = b1*u2 - b2*u1
    sF13 = b1*u3 - b3*u1
    sF23 = b2*u3 - b3*u2
    
    # Tmunu mag (contravarient)
    T00_mag = bsq*u0*u0 - b0*b0 + 0.5*bsq*gcon[0][0]
    T01_mag = bsq*u0*u1 - b0*b1 + 0.5*bsq*gcon[0][1]
    T02_mag = bsq*u0*u2 - b0*b2 + 0.5*bsq*gcon[0][2]
    T03_mag = bsq*u0*u3 - b0*b3 + 0.5*bsq*gcon[0][3]
    
    T11_mag = bsq*u1*u1 - b1*b1 + 0.5*bsq*gcon[1][1]
    T12_mag = bsq*u1*u2 - b1*b2 + 0.5*bsq*gcon[1][2]
    T13_mag = bsq*u1*u3 - b1*b3 + 0.5*bsq*gcon[1][3]

    T22_mag = bsq*u2*u2 - b2*b2 + 0.5*bsq*gcon[2][2]
    T23_mag = bsq*u2*u3 - b2*b3 + 0.5*bsq*gcon[2][3]

    T33_mag = bsq*u3*u3 - b3*b3 + 0.5*bsq*gcon[3][3]

    # Tmunu hd (contravarient)
    T00_hd = w*u0*u0 + pgas*gcon[0][0]
    T01_hd = w*u0*u1 + pgas*gcon[0][1]
    T02_hd = w*u0*u2 + pgas*gcon[0][2]
    T03_hd = w*u0*u3 + pgas*gcon[0][3]
    
    T11_hd = w*u1*u1 + pgas*gcon[1][1]
    T12_hd = w*u1*u2 + pgas*gcon[1][2]
    T13_hd = w*u1*u3 + pgas*gcon[1][3]

    T22_hd = w*u2*u2 + pgas*gcon[2][2]
    T23_hd = w*u2*u3 + pgas*gcon[2][3]

    T33_hd = w*u3*u3 + pgas*gcon[3][3]
                           
    # mass flux
    rhou1 = rho*u1
    
    ############################################################################################
    # make a new hdf5 file
    fout = h5py.File(fileout,'w')
    try:
        fout.create_dataset('t',data=fin['t'][()])
        
        # Header
        grp = fout.create_group('header')    
        grp.create_dataset('bhspin',data=spin)
        grp.create_dataset('file_number',data=fin['header']['file_number'][()])
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
        grp.create_dataset('r',  data= np.mean(r, axis=2))
        grp.create_dataset('th', data= np.mean(th, axis=2))
        grp.create_dataset('ph', data= np.zeros((NX,NY)))
        
        # Primitive Quantities
        grp = fout.create_group('quants')
        grp.create_dataset('rho',  data= np.mean(rho,  axis=2))
        grp.create_dataset('uint', data= np.mean(uint, axis=2))
        
        grp.create_dataset('B1', data= np.mean(B1, axis=2))
        grp.create_dataset('B2', data= np.mean(B2, axis=2))
        grp.create_dataset('B3', data= np.mean(B3, axis=2))
        
        grp.create_dataset('U1', data= np.mean(u1_velr, axis=2))
        grp.create_dataset('U2', data= np.mean(u2_velr, axis=2))
        grp.create_dataset('U3', data= np.mean(u3_velr, axis=2))    
        
        # Derived Quantities
        grp.create_dataset('lorentz', data= np.nanmean(lorentz, axis=2))
        grp.create_dataset('bsq',     data= np.nanmean(bsq,     axis=2))
        grp.create_dataset('sigma',   data= np.nanmean(sigma,   axis=2))    
        grp.create_dataset('beta',    data= np.nanmean(beta,    axis=2))    
        grp.create_dataset('Tgas',    data= np.nanmean(Tgas,    axis=2))    
        grp.create_dataset('betainv', data= np.nanmean(betainv, axis=2))                

        grp.create_dataset('b0', data= np.nanmean(b0, axis=2))
        grp.create_dataset('b1', data= np.nanmean(b1, axis=2))
        grp.create_dataset('b2', data= np.nanmean(b2, axis=2))
        grp.create_dataset('b3', data= np.nanmean(b3, axis=2))
        
        grp.create_dataset('u0', data= np.nanmean(u0, axis=2))
        grp.create_dataset('u1', data= np.nanmean(u1, axis=2))
        grp.create_dataset('u2', data= np.nanmean(u2, axis=2))
        grp.create_dataset('u3', data= np.nanmean(u3, axis=2))

        grp.create_dataset('sF12',  data= np.nanmean(sF12,  axis=2))    
        grp.create_dataset('sF13',  data= np.nanmean(sF13,  axis=2))    
        grp.create_dataset('sF23',  data= np.nanmean(sF23,  axis=2))    
        
        grp.create_dataset('rhou1', data= np.nanmean(rhou1, axis=2))    
        
        grp.create_dataset('T00_mag',  data= np.nanmean(T00_mag,  axis=2))    
        grp.create_dataset('T01_mag',  data= np.nanmean(T01_mag,  axis=2))    
        grp.create_dataset('T02_mag',  data= np.nanmean(T02_mag,  axis=2))    
        grp.create_dataset('T03_mag',  data= np.nanmean(T03_mag,  axis=2))    
        grp.create_dataset('T11_mag',  data= np.nanmean(T11_mag,  axis=2))    
        grp.create_dataset('T12_mag',  data= np.nanmean(T12_mag,  axis=2))    
        grp.create_dataset('T13_mag',  data= np.nanmean(T13_mag,  axis=2))    
        grp.create_dataset('T22_mag',  data= np.nanmean(T22_mag,  axis=2))    
        grp.create_dataset('T23_mag',  data= np.nanmean(T23_mag,  axis=2))    
        grp.create_dataset('T33_mag',  data= np.nanmean(T33_mag,  axis=2))    
                                                        
        grp.create_dataset('T00_hd',  data= np.nanmean(T00_hd,  axis=2))    
        grp.create_dataset('T01_hd',  data= np.nanmean(T01_hd,  axis=2))    
        grp.create_dataset('T02_hd',  data= np.nanmean(T02_hd,  axis=2))    
        grp.create_dataset('T03_hd',  data= np.nanmean(T03_hd,  axis=2))    
        grp.create_dataset('T11_hd',  data= np.nanmean(T11_hd,  axis=2))    
        grp.create_dataset('T12_hd',  data= np.nanmean(T12_hd,  axis=2))    
        grp.create_dataset('T13_hd',  data= np.nanmean(T13_hd,  axis=2))    
        grp.create_dataset('T22_hd',  data= np.nanmean(T22_hd,  axis=2))    
        grp.create_dataset('T23_hd',  data= np.nanmean(T23_hd,  axis=2))    
        grp.create_dataset('T33_hd',  data= np.nanmean(T33_hd,  axis=2))    
      
    except:
        print("Error writing averaged hdf5 file ", fileout, "!")   

    fin.close()
    fout.close()
    
    return

if __name__=='__main__':
    main()
