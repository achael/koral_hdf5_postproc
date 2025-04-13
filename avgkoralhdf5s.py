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
                   
def phireduce_hdf5(filein, fileout, reducetype='avg', phiidx=PHIIDX, metric_avg=METRIC,tmin=TMIN1,tmax=TMAX, rerun=True, verbose=True):
    """Either average or slice a koral hdf5 file"""
    if verbose: print('averaging hdf5 ', filein, '....')
    if metric_avg not in ['KS','BL']:
        raise Exception("metric_avg must be 'KS' or 'BL'")    
    if (not rerun) and os.path.exists(fileout):
        return
    
    # reduce function
    def pr(data):
        return phireduce(data, reducetype=reducetype, phiidx=phiidx)  
           
    # load data     
    try:
        fin = h5py.File(filein,'r')

        # time - do not process if out of range
        time = fin['t'][()]
        if time<tmin or time>tmax:
            fin.close()
            return
                     
        # get info from the header
        spin = fin['header']['bhspin'][()].astype('f')
        horiz = 1 + np.sqrt(1-spin**2)
        metric_run = fin['header']['metric_run'][()]
        metric_out = fin['header']['metric_out'][()]
        has_radiation = bool(fin['header']['has_radiation'][()])
        has_electrons = bool(fin['header']['has_electrons'][()])

        masssolar = fin['header']['units']['M_bh'][()]
        mass_cgs = masssolar * MSUN_CGS
        l_unit = fin['header']['units']['L_unit'][()] # length code2cgs = GM/c^2
        u_unit = fin['header']['units']['U_unit'][()] # energy density code2cgs = c^8/G^3M^2  
        m_unit = fin['header']['units']['M_unit'][()] # mass density code2cgs = c^6/G^3M^2  
                
        if verbose:
            print(f"M:{mass_cgs}, L:{l_unit}, U:{u_unit}, rho: {m_unit}")
                     
        if not isinstance(metric_run, str): metric_run = metric_run.decode('utf-8')
        if not isinstance(metric_out, str): metric_out = metric_out.decode('utf-8') 
        if metric_out not in ['KS','BL']:
            raise Exception("metric_out of KORAL h5 file must be KS or BL!")

        NX = fin['header']['n1'][()]
        NY = fin['header']['n2'][()]
        NZ = fin['header']['n3'][()]
        gamma_head = fin['header']['gam'][()]

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
        
        if has_radiation:
            erad = fin['quants']['erad'][:,:,:]
            try:
                nphot = fin['quants']['nphot'][:,:,:]
                has_photons = True
            except:
                nphot = np.zeros(erad.shape)
                has_photons = False
                
            ur1_velr = fin['quants']['F1'][:,:,:]
            ur2_velr = fin['quants']['F2'][:,:,:]
            ur3_velr = fin['quants']['F3'][:,:,:]
                                
        if has_electrons:
            TeK = fin['quants']['te'][:,:,:]
            TiK = fin['quants']['ti'][:,:,:]
            gamma_adiab = fin['quants']['gammagas'][:,:,:]
        else:
            gamma_adiab = gamma_head
    except: 
        print("Error reading h5 file ", filein, "!")
            
    # Calculate derived quantites
    # Metric
    if metric_avg=="KS": 
        gcon = gconKS(spin,r,th)
        lower = lowerKS
        conv_vel = conv_vel_KS
        trans_cov = trans_cov_bl2ks
        invconv_vel = invconv_vel_KS
    elif metric_avg=="BL": 
        gcon=gconBL(spin,r,th)
        lower = lowerBL
        conv_vel = conv_vel_BL
        trans_cov = trans_cov_ks2bl 
        invconv_vel = invconv_vel_BL                 
    # get 4-velocity 
    (u0,u1,u2,u3) = conv_vel(u1_velr, u2_velr, u3_velr, spin, r, th)
    (u0_l,u1_l,u2_l,u3_l) = lower(u0, u1, u2, u3, spin, r, th)

    if has_radiation:
        (ur0,ur1,ur2,ur3) = conv_vel(ur1_velr, ur2_velr, ur3_velr, spin, r, th)
        (ur0_l,ur1_l,ur2_l,ur3_l) = lower(ur0, ur1, ur2, ur3, spin, r, th)
            
    # get magnetic field 4-vector and b^2
    b0 = u1_l*B1 + u2_l*B2 + u3_l*B3
    b1 = (B1 + b0*u1)/u0
    b2 = (B2 + b0*u2)/u0
    b3 = (B3 + b0*u3)/u0

    (b0_l,b1_l,b2_l,b3_l) = lower(b0, b1, b2, b3, spin, r, th)
    bsq = b0*b0_l + b1*b1_l + b2*b2_l + b3*b3_l
    
    # change coordinates
    if metric_out != metric_avg:
        (u0,u1,u2,u3) = trans_cov(u0, u1, u2, u3, spin, r, th)
        (u0_l, u1_l, u2_l, u3_l) = lower(u0,u1,u2,u3,spin,r,th)
        u1_velr, u2_velr, u3_velr = invconv_vel(u0, u1, u2, u3, spin, r, th)
                
        (b0,b1,b2,b3) = trans_cov(b0, b1, b2, b3, spin, r, th)
        (b0_l, b1_l, b2_l, b3_l) = lower(b0,b1,b2,b3,spin,r,th)
        
        if has_radiation:
            (ur0,ur1,ur2,ur3) = trans_cov(ur0, ur1, ur2, ur3, spin, r, th)
            (ur0_l, ur1_l, ur2_l, ur3_l) = lower(ur0,ur1,ur2,ur3,spin,r,th)
            ur1_velr, ur2_velr, ur3_velr = invconv_vel(ur0, ur1, ur2, ur3, spin, r, th)
                    
    # (re) derive B-field 3 vector in case coordinates changed
    B1 = b1*u0 - b0*u1
    B2 = b2*u0 - b0*u2
    B3 = b3*u0 - b0*u3  
 
    # plasma parameters
    pgas = (gamma_adiab - 1.)*uint # pressure
    w = (rho + pgas + uint)     # enthalpy
    TgasK = (pgas / rho) * (MU_GAS / TPFAC) # gas temperature  in Kelvin 
         
    # radiation derived quantities
    if has_radiation:

        # fluid frame radiation energy density and fluid frame  photon number
        erad_hat = np.zeros(erad.shape)  
                
        urcon = (ur0, ur1, ur2, ur3)
        ucov = (u0_l, u1_l, u2_l, u3_l)
        for i in range(4):
            for j in range(4):
                erad_hat += ((4./3.)*erad*urcon[i]*urcon[j] + (1./3.)*erad*gcon[i][j])*ucov[i]*ucov[j]

        # fluid frame photon number
        if has_photons:
            nphot_hat = np.zeros(nphot.shape) 
            for i in range(4):
                nphot_hat += -nphot*urcon[i]*ucov[i]
                
            # radiation temperature in the fluid frame  
            TradK = erad_hat/(2.7012*nphot_hat) * (mass_cgs*C_CGS*C_CGS / KBOLTZ_CGS) # in Kelvin TODO check
        else:
            TradK = ((erad_hat*u_unit / A_RAD_CGS))**(0.25)  # in Kelvin
            nphot_hat = (A_RAD_CGS * (TradK**3) / (2.70118*KBOLTZ_CGS)) * (mass_cgs*C_CGS*C_CGS / u_unit)
                                       

        # opacities
        # u_unit = c^8/G^3 M^2 is the trnasformation for energy density from code --> cgs
        rhocgs =  rho * (u_unit / (C_CGS*C_CGS))
        Bmagcgs = np.sqrt(bsq) * np.sqrt(4*np.pi*u_unit)
            
        if not has_electrons: 
            TeK = TgasK     
            
        # opacities (compute in cgs from koralopacities.py and and return to code units)
        # TODO include scattering?
        (synemisopac, synabsorbopac) = synopac(rhocgs, TeK, TradK, Bmagcgs)
        (ffemisopac, ffabsorbopac) = ffopac(rhocgs, TeK, TradK)
        comptabsorbopac = comptopac(rhocgs, TeK, TradK, kn=KLEINNISHINA)

        # return to code units
        synemisopac *= l_unit
        synabsorbopac *= l_unit
        ffemisopac *= l_unit
        ffabsorbopac *= l_unit
        comptabsorbopac *= l_unit
        
        # blackbody energy density (code units)
        BBenergy = A_RAD_CGS*(TeK**4) / u_unit # blackbody energy in code units   
                        
    if has_electrons:

        pi = (rho/MU_I)*TPFAC*TiK  # in code units
        pe = (rho/MU_E)*TPFAC*TeK  # in code units
                
        # heating functions (unitless)
        deltaeK = deltaeKawazura(2*pi/bsq, TeK/TiK)
        deltaeZ = deltaeZhdankin(TEFAC*TeK, TPFAC*TiK/MU_I,  MRATIO) 
        

                     
    ###########################################################################################
    # make the output hdf5 file
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
        grp.create_dataset('metric_out',data=metric_avg) 
        grp.create_dataset('metric_run',data=fin['header']['metric_run'][()])
        grp.create_dataset('n1',data=NX) 
        grp.create_dataset('n2',data=NY) 
        grp.create_dataset('n3',data=1)    #phi-averaged
        grp.create_dataset('ndim',data=2)  #phi-averaged
        grp.create_dataset('problem_number',data=fin['header']['problem_number'][()])     
        grp.create_dataset('version',data=fin['header']['version'][()])
        
        grp.copy(fin['header']['units'], grp)
        grp.copy(fin['header']['geom'], grp)
        
        # Grid
        grp = fout.create_group('grid_out')
        grp.create_dataset('r',  data= pr(r))
        grp.create_dataset('th', data= pr(th))
        
        if reducetype=='slice':
            phidata = ph[:,:,phiidx]
        else:
            phidata = np.zeros((NX,NY))
        grp.create_dataset('ph', data=phidata)
        
    except: 
        print("Error creating averaged hdf5 file ", fileout, "!")   
        #os.remove(fileout)    
    
    # write averaged and derived quantities
    try: 
        # Primitive Quantities
        grp = fout.create_group('quants')
        grp.create_dataset('rho',  data= pr(rho))
        grp.create_dataset('uint', data= pr(uint))
        
        grp.create_dataset('B1', data= pr(B1))
        grp.create_dataset('B2', data= pr(B2))
        grp.create_dataset('B3', data= pr(B3))
        
        grp.create_dataset('U1', data= pr(u1_velr))
        grp.create_dataset('U2', data= pr(u2_velr))
        grp.create_dataset('U3', data= pr(u3_velr))  

        # Derived Quantities
        # we don't need the lorentz factor for axisymmetric metric, we can get it from u^0
        #grp.create_dataset('lorentz', data= pr(lorentz))
        grp.create_dataset('pgas',     data= pr(pgas)) # redundant for MHD but useful with variable adiabatic index
        grp.create_dataset('bsq',     data= pr(bsq))
        grp.create_dataset('sigma',   data= pr(bsq / rho))
        grp.create_dataset('sigmaw',  data= pr(bsq / w))
        grp.create_dataset('Tgas',    data= pr(TgasK)) # TODO in kelvin or not? 
        grp.create_dataset('beta',    data= pr(pgas / (0.5*bsq)))        
        grp.create_dataset('betainv', data= pr((0.5*bsq) / pgas))

        grp.create_dataset('absB1', data= pr(np.abs(B1)))
        grp.create_dataset('absB2', data= pr(np.abs(B2)))
        grp.create_dataset('absB3', data= pr(np.abs(B3)))
                        
        grp.create_dataset('b0', data= pr(b0))
        grp.create_dataset('b1', data= pr(b1))
        grp.create_dataset('b2', data= pr(b2))
        grp.create_dataset('b3', data= pr(b3))
        
        grp.create_dataset('u0', data= pr(u0))
        grp.create_dataset('u1', data= pr(u1))
        grp.create_dataset('u2', data= pr(u2))
        grp.create_dataset('u3', data= pr(u3))

        # spatial components of Maxwell (contravariant)
        grp.create_dataset('sF12',  data= pr(b1*u2 - b2*u1)) 
        grp.create_dataset('sF13',  data= pr(b1*u3 - b3*u1))
        grp.create_dataset('sF23',  data= pr(b2*u3 - b3*u2))
        
        # Tmunu mag (contravarient)        
        grp.create_dataset('T00_mag',  data= pr(bsq*u0*u0 - b0*b0 + 0.5*bsq*gcon[0][0]))
        grp.create_dataset('T01_mag',  data= pr(bsq*u0*u1 - b0*b1 + 0.5*bsq*gcon[0][1]))
        grp.create_dataset('T02_mag',  data= pr(bsq*u0*u2 - b0*b2 + 0.5*bsq*gcon[0][2]))
        grp.create_dataset('T03_mag',  data= pr(bsq*u0*u3 - b0*b3 + 0.5*bsq*gcon[0][3]))
        grp.create_dataset('T11_mag',  data= pr(bsq*u1*u1 - b1*b1 + 0.5*bsq*gcon[1][1]))
        grp.create_dataset('T12_mag',  data= pr(bsq*u1*u2 - b1*b2 + 0.5*bsq*gcon[1][2]))
        grp.create_dataset('T13_mag',  data= pr(bsq*u1*u3 - b1*b3 + 0.5*bsq*gcon[1][3]))
        grp.create_dataset('T22_mag',  data= pr(bsq*u2*u2 - b2*b2 + 0.5*bsq*gcon[2][2]))
        grp.create_dataset('T23_mag',  data= pr(bsq*u2*u3 - b2*b3 + 0.5*bsq*gcon[2][3]))
        grp.create_dataset('T33_mag',  data= pr(bsq*u3*u3 - b3*b3 + 0.5*bsq*gcon[3][3]))

        # Tmunu mat (contravarient)                                                                 
        grp.create_dataset('T00_hd',  data= pr(w*u0*u0 + pgas*gcon[0][0]))
        grp.create_dataset('T01_hd',  data= pr(w*u0*u1 + pgas*gcon[0][1]))
        grp.create_dataset('T02_hd',  data= pr(w*u0*u2 + pgas*gcon[0][2]))
        grp.create_dataset('T03_hd',  data= pr(w*u0*u3 + pgas*gcon[0][3]))
        grp.create_dataset('T11_hd',  data= pr(w*u1*u1 + pgas*gcon[1][1]))
        grp.create_dataset('T12_hd',  data= pr(w*u1*u2 + pgas*gcon[1][2]))
        grp.create_dataset('T13_hd',  data= pr(w*u1*u3 + pgas*gcon[1][3]))
        grp.create_dataset('T22_hd',  data= pr(w*u2*u2 + pgas*gcon[2][2]))
        grp.create_dataset('T23_hd',  data= pr(w*u2*u3 + pgas*gcon[2][3]))
        grp.create_dataset('T33_hd',  data= pr(w*u3*u3 + pgas*gcon[3][3]))
      
        # rho- weighted quantities
        grp.create_dataset('rhosq', data= pr(rho*rho))        
        grp.create_dataset('rhobsq', data= pr(rho*bsq))        
        grp.create_dataset('rhouint', data= pr(rho*uint))
        grp.create_dataset('rhopgas', data= pr(rho*pgas))
        grp.create_dataset('rhoscaleheight', data= pr(rho*np.abs(th - 0.5*np.pi)))
                
        grp.create_dataset('rhou0', data= pr(rho*u0))
        grp.create_dataset('rhou1', data= pr(rho*u1))
        grp.create_dataset('rhou2', data= pr(rho*u2))
        grp.create_dataset('rhou3', data= pr(rho*u3))
     
        grp.create_dataset('rhoabsB1', data= pr(rho*np.abs(B1)))
        grp.create_dataset('rhoabsB2', data= pr(rho*np.abs(B2)))
        grp.create_dataset('rhoabsB3', data= pr(rho*np.abs(B3)))
              
        grp.create_dataset('rhoB1', data= pr(rho*B1))
        grp.create_dataset('rhoB2', data= pr(rho*B2))
        grp.create_dataset('rhoB3', data= pr(rho*B3))

        grp.create_dataset('rhob0', data= pr(rho*b0))        
        grp.create_dataset('rhob1', data= pr(rho*b1))
        grp.create_dataset('rhob2', data= pr(rho*b2))
        grp.create_dataset('rhob3', data= pr(rho*b3))        

        if has_radiation:
            grp.create_dataset('erad',  data= pr(erad))  # radiation frame
            grp.create_dataset('nphot', data= pr(nphot)) # radiation frame
         
            grp.create_dataset('F1', data= pr(ur1_velr))
            grp.create_dataset('F2', data= pr(ur2_velr))
            grp.create_dataset('F3', data= pr(ur3_velr))  

            grp.create_dataset('ur0', data= pr(ur0))
            grp.create_dataset('ur1', data= pr(ur1))
            grp.create_dataset('ur2', data= pr(ur2))
            grp.create_dataset('ur3', data= pr(ur3))
                
            grp.create_dataset('T00_rad',  data= pr((4./3.)*erad*ur0*ur0 + (1./3.)*erad*gcon[0][0]))
            grp.create_dataset('T01_rad',  data= pr((4./3.)*erad*ur0*ur1 + (1./3.)*erad*gcon[0][1]))
            grp.create_dataset('T02_rad',  data= pr((4./3.)*erad*ur0*ur2 + (1./3.)*erad*gcon[0][2]))
            grp.create_dataset('T03_rad',  data= pr((4./3.)*erad*ur0*ur3 + (1./3.)*erad*gcon[0][3]))
            grp.create_dataset('T11_rad',  data= pr((4./3.)*erad*ur1*ur1 + (1./3.)*erad*gcon[1][1]))
            grp.create_dataset('T12_rad',  data= pr((4./3.)*erad*ur1*ur2 + (1./3.)*erad*gcon[1][2]))
            grp.create_dataset('T13_rad',  data= pr((4./3.)*erad*ur1*ur3 + (1./3.)*erad*gcon[1][3]))
            grp.create_dataset('T22_rad',  data= pr((4./3.)*erad*ur2*ur2 + (1./3.)*erad*gcon[2][2]))
            grp.create_dataset('T23_rad',  data= pr((4./3.)*erad*ur2*ur3 + (1./3.)*erad*gcon[2][3]))
            grp.create_dataset('T33_rad',  data= pr((4./3.)*erad*ur3*ur3 + (1./3.)*erad*gcon[3][3]))

            # fluid frame quantities
            grp.create_dataset('erad_hat',  data= pr(erad_hat))        
            grp.create_dataset('nphot_hat', data= pr(nphot_hat))
            grp.create_dataset('Trad',   data= pr(TradK)) 

            # opacities
            grp.create_dataset('opac_syn', data= pr(synabsorbopac))   
            grp.create_dataset('opac_ff', data= pr(ffabsorbopac))   
            #grp.create_dataset('opac_compt', data= pr(comptabsorbopac))   # TODO don't double count with opac_compt
            
            grp.create_dataset('emis_syn', data= pr(-BBenergy*synemisopac))   
            grp.create_dataset('emis_ff', data= pr(-BBenergy*ffemisopac))    
            grp.create_dataset('emis_compt', data= pr(erad_hat*comptabsorbopac))  # should be negative
            
            # rho weighted quantites (?)
            # TODO do we want ebar weighted quantities?             
            grp.create_dataset('rhoehat', data=pr(rho*erad_hat))
            grp.create_dataset('rhonhat', data=pr(rho*nphot_hat))
                            
            grp.create_dataset('rhoTrad', data=pr(rho*TradK))
            
            grp.create_dataset('rhour0', data= pr(rho*ur0))
            grp.create_dataset('rhour1', data= pr(rho*ur1))
            grp.create_dataset('rhour2', data= pr(rho*ur2))
            grp.create_dataset('rhour3', data= pr(rho*ur3))
                                                        
        if has_electrons: 
            grp.create_dataset('ti', data= pr(TiK)) # TODO in kelvin or not? 
            grp.create_dataset('te', data= pr(TeK)) # TODO in kelvin or not? 
            grp.create_dataset('gammagas', data=pr(gamma_adiab))
            
            grp.create_dataset('pi', data= pr(pi))
            grp.create_dataset('pe', data= pr(pe))   

            # derived quantities
            grp.create_dataset('deltaeK', data=pr(deltaeK))
            grp.create_dataset('deltaeZ', data=pr(deltaeZ))
                        
            # rho weighted quantites (?)
            grp.create_dataset('rhope', data=pr(rho*pe))
            grp.create_dataset('rhopi', data=pr(rho*pi))                          
            grp.create_dataset('rhogammagas', data=pr(rho*gamma_adiab))
                              
    except:
        print("Error writing to averaged hdf5 file ", fileout, "!")   
        #os.remove(fileout)
        
    fin.close()
    fout.close()
    
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
