import pylab
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from scipy.interpolate import griddata
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from metricKS import *
import h5py

def read_phiavg_hdf5(filein, verbose=True, metric=METRIC_OUT):
    """ general read file from tavg_koral_hdf5s_BLallquants.py"""
    
    if verbose: print('reading hdf5 ', filein, '....')
    
    # load data     
    fin = h5py.File(filein,'r')

    # get info from the header
    spin = fin['header']['bhspin'][()].astype('f')
    horiz = 1 + np.sqrt(1-spin**2)
    metric_run = fin['header']['metric_run'][()]
    metric_out = fin['header']['metric_out'][()]
    if not isinstance(metric_run, str): metric_run = metric_run.decode('utf-8')
    if not isinstance(metric_out, str): metric_out = metric_out.decode('utf-8')
    if metric_out != metric:
        raise Exception("output metric %s must be %s!"%(metric_out,metric))
    gamma_adiab = fin['header']['gam'][()]

    # output dictionary
    outdict = {}

    # header
    outdict['spin'] = spin
    outdict['metric'] = metric_out
    outdict['gamma_adiab'] = gamma_adiab
      
    # coords  
    outdict['r'] = fin['grid_out']['r'][:]
    outdict['th'] = fin['grid_out']['th'][:]
        
    # get simulation primitive variables
    outdict['rho'] = fin['quants']['rho'][:]
    outdict['uint'] = fin['quants']['uint'][:]

    outdict['U1'] = fin['quants']['U1'][:]
    outdict['U2'] = fin['quants']['U2'][:]
    outdict['U3'] = fin['quants']['U3'][:]

    outdict['B1'] = fin['quants']['B1'][:]
    outdict['B2'] = fin['quants']['B2'][:]
    outdict['B3'] = fin['quants']['B3'][:]
    
    # get simulation phiaveraged derived quantities
    outdict['lorentz'] = fin['quants']['lorentz'][:]
    outdict['bsq'] = fin['quants']['bsq'][:]
    outdict['sigma'] = fin['quants']['sigma'][:]
    outdict['beta'] = fin['quants']['beta'][:]
    outdict['Tgas'] = fin['quants']['Tgas'][:]
    outdict['betainv'] = fin['quants']['betainv'][:]    

    outdict['b0'] = fin['quants']['b0'][:]
    outdict['b1'] = fin['quants']['b1'][:]
    outdict['b2'] = fin['quants']['b2'][:]
    outdict['b3'] = fin['quants']['b3'][:]

    outdict['u0'] = fin['quants']['u0'][:]
    outdict['u1'] = fin['quants']['u1'][:]
    outdict['u2'] = fin['quants']['u2'][:]
    outdict['u3'] = fin['quants']['u3'][:]

    outdict['sF12'] = fin['quants']['sF12'][:]
    outdict['sF13'] = fin['quants']['sF13'][:]
    outdict['sF23'] = fin['quants']['sF23'][:]
        
    outdict['rhou1'] = fin['quants']['rhou1'][:]
    
    outdict['T00_mag'] = fin['quants']['T00_mag'][:]
    outdict['T01_mag'] = fin['quants']['T01_mag'][:]
    outdict['T02_mag'] = fin['quants']['T02_mag'][:]
    outdict['T03_mag'] = fin['quants']['T03_mag'][:]
    outdict['T11_mag'] = fin['quants']['T11_mag'][:]
    outdict['T12_mag'] = fin['quants']['T12_mag'][:]
    outdict['T13_mag'] = fin['quants']['T13_mag'][:]
    outdict['T22_mag'] = fin['quants']['T22_mag'][:]
    outdict['T23_mag'] = fin['quants']['T23_mag'][:]
    outdict['T33_mag'] = fin['quants']['T33_mag'][:]    

    outdict['T00_hd'] = fin['quants']['T00_hd'][:]
    outdict['T01_hd'] = fin['quants']['T01_hd'][:]
    outdict['T02_hd'] = fin['quants']['T02_hd'][:]
    outdict['T03_hd'] = fin['quants']['T03_hd'][:]
    outdict['T11_hd'] = fin['quants']['T11_hd'][:]
    outdict['T12_hd'] = fin['quants']['T12_hd'][:]
    outdict['T13_hd'] = fin['quants']['T13_hd'][:]
    outdict['T22_hd'] = fin['quants']['T22_hd'][:]
    outdict['T23_hd'] = fin['quants']['T23_hd'][:]
    outdict['T33_hd'] = fin['quants']['T33_hd'][:]    
                            
    # close file
    fin.close()
    
    return outdict
    
                   
def unpack_hdf5_blallquants(filein):
    """read a phi-averaged hdf5 file 
       then do some post-processing for derived quantities
       modified for 'BL_allquants version"""
       

    datadict = read_phiavg_hdf5(filein, verbose=True, metric=METRIC_OUT)
    outdat = {}

    NX = datadict['r'].shape[0]
    NY = datadict['r'].shape[1]
    
    spin = datadict['spin']
    outdat['spin'] = spin
    outdat['metric'] = datadict['metric']
    outdat['gamma_adiab'] = datadict['gamma_adiab']
        
    r = datadict['r']
    th = datadict['th']
    Delta = r**2 - 2*r + spin**2
    
    outdat['r'] = r
    outdat['th'] = th
    if METRIC_OUT=='KS':
        outdat['gdet'] = gdetKS(spin, r, th) 
    elif METRIC_OUT=='BL':  
        outdat['gdet'] = gdetBL(spin, r, th)  
                 
    ####################################################################################
    # get directly averaged quantities     
    outdat['rho'] = datadict['rho']
    outdat['uint'] = datadict['uint']
    outdat['lorentz'] = datadict['lorentz']
    outdat['bsq']  = datadict['bsq']
    outdat['sigma']  = datadict['sigma']
    outdat['beta']  = datadict['beta']        
    outdat['betainv']  = datadict['betainv']
    outdat['Tgas']  = datadict['Tgas']

    B1 = datadict['B1']
    B2 = datadict['B2']
    B3 = datadict['B3']
    outdat['Bcon'] = (0,B1,B2,B3)
                    
    ####################################################################################    
    
    # derived quantities
    
    # normalized 4-velocity 
    # TODO is this what we want for plotting v^phi
    if METRIC_OUT=='KS':
        (u0,u1,u2,u3) = conv_vel_KS(datadict['U1'], datadict['U2'], datadict['U3'], spin, r, th)
        (u0_l,u1_l,u2_l,u3_l) = lowerKS(u0,u1,u2,u3,spin,r,th)
    elif METRIC_OUT=='BL':  
        (u0,u1,u2,u3) = conv_vel_BL(datadict['U1'], datadict['U2'], datadict['U3'], spin, r, th)
        (u0_l,u1_l,u2_l,u3_l) = lowerBL(u0,u1,u2,u3,spin,r,th)
                
    # get Aphi contour integrating in theta
    gdet = outdat['gdet']
    dth = 0.5*(th[:,2:] - th[:,:-2]) # approx dtheta        
    integrand = (B1[:,1:-1]*gdet[:,1:-1])*dth
    aphi_th = np.zeros(r.shape)
    for i in range(NY-2):
        if i == 0: 
            aphi_th[:,0] = (integrand*dth)[:,0]
            aphi_th[:,0+1] = (integrand*dth)[:,0]
        else: 
            aphi_th[:,i+1] = aphi_th[:,i] + (integrand*dth)[:,i]
    aphi_th[:,-1] = aphi_th[:,-2]

    # get Aphi contour integrating in theta
    # this gives "worse" results
    dr = 0.5*(r[2:,:] - r[:-2,:]) # approx dtheta        
    integrand = -(B2[1:-1,:]*gdet[1:-1,:])*dr
    aphi_r = np.zeros(r.shape)
    for i in range(NX-2):
        if i == 0: 
            aphi_r[0,:] = (integrand*dr)[0,:]
            aphi_r[0+1,:] = (integrand*dr)[0,:]
        else: 
            aphi_r[i+1,:] = aphi_r[i,:] + (integrand*dr)[i,:]
    aphi_r[-1,:] = aphi_r[-2,:]    
    
    # fieldline angular speeds from averaged contravarient Maxwell
    omega_r  = datadict['sF13']/datadict['B1']
    omega_th = datadict['sF23']/datadict['B2']
    
    
    # energy fluxes
    # TODO general lower function for tensors
    if METRIC_OUT=='KS':
        g = gcovKS(spin,r,th)
    elif METRIC_OUT=='BL':
        g = gcovBL(spin,r,th)    
        
    fe_mag = -1*(g[0][0]*datadict['T01_mag'] + g[0][1]*datadict['T11_mag'] + g[0][2]*datadict['T12_mag'] + g[0][3]*datadict['T13_mag'])
    fe_hd  = -1*(g[0][0]*datadict['T01_hd']  + g[0][1]*datadict['T11_hd']  + g[0][2]*datadict['T12_hd']  + g[0][3]*datadict['T13_hd'])
    
    # b four vector and 
    b0 = B1*u1_l + B2*u2_l + B3*u3_l
    b1 = (B1 + b0*u1)/u0
    b2 = (B2 + b0*u2)/u0
    b3 = (B3 + b0*u3)/u0        
    (b0_l,b1_l,b2_l,b3_l) = lowerBL(b0,b1,b2,b3,spin,r,th)
    bsq = b0*b0_l + b1*b1_l + b2*b2_l + b3*b3_l
    
    
    #################### calculate energy flux from averaged fields in various ways###############
    #fe_mag_prims = -1*(bsq*u1*u0_l - b1*b0_l)
    fe_mag_prims = -B1*B3*omega_r*Delta*np.sin(th)**2
    
    # from all components of averaged star F
    datshape = np.array(datadict['sF12'].shape)
    zerovec = np.zeros(datshape)
    sFcon = np.zeros(np.array(((4,4),datshape)).flatten())
    for i in range(4):
        for j in range(i,4):
            if i == j: 
                sFcon[i,j] = zerovec
            elif i==0:
                sFcon[i,j] = -1*datadict['B%d'%j]
            else:
                sFcon[i,j] = datadict['sF%d%d'%(i,j)]
            sFcon[j,i] = -1*sFcon[i][j]
    sFcov = lower22(sFcon, g)
    
    fe_mag_starf = zerovec
    for i in range(4):
        fe_mag_starf += sFcon[1,i]*sFcov[i,0]
    
    # nelecting non axisym parts
    #fe_mag_starf = -Delta*np.sin(th)**2 * sFcon[1][3] * sFcon[3][0]
    #fe_mag_starf = sFcon[1][3]*sFcov[3][0]
    # non axisym parts only 
    #femag_starf = sFcon[1][2]*sFcov[2][0]
    ###################################
            
    # package in dictionary
    outdat['ucon'] = (u0,u1,u2,u3)
    outdat['Aphi_th'] = aphi_th
    outdat['Aphi_r'] = aphi_r
    outdat['omega_th'] = omega_th
    outdat['omega_r'] = omega_r
        
    outdat['fe_mag'] = fe_mag
    outdat['fe_hd'] = fe_hd

    outdat['fe_mag_prims'] = fe_mag_prims
    outdat['fe_mag_starf'] = fe_mag_starf 
    return outdat

def my_griddata(datdict, field, xlim=xlim, zlim1=zlim1, zlim2=zlim2, ngrid=500j, verbose=True, reggrid=True):
  
    spin = datdict['spin']
    horiz = 1 + np.sqrt(1-spin**2)    
    omegaH = spin / (2*horiz)
    if spin==0: 
        omegaISCO = 1/(6**1.5)
        omegaH = omegaISCO
        #ospin = 0.7
        #omegaH= ospin / (2 + 2*np.sqrt(1-ospin**2)  )
           
    r = datdict['r']
    th = datdict['th']
    
    # put on a regular grid or keep simulation grid? 
    if reggrid:
        points=((r*np.sin(th)).flatten(), (r*np.cos(th)).flatten())    
        grid_z, grid_x = np.mgrid[zlim1:zlim2:ngrid,  0:xlim:ngrid]  
        grid_r = np.sqrt(grid_z**2 + grid_x**2) 
        def my_griddata2(data):
            grid_data = griddata(points, data.flatten(), (grid_x, grid_z), method='cubic', fill_value=np.nan) 
            grid_data = np.ma.masked_where(np.isnan(grid_data), grid_data)
            return grid_data
    else:
        grid_x = (r*np.sin(th))
        grid_z = (r*np.cos(th))  
        grid_r = np.sqrt(grid_z**2 + grid_x**2) 
        def my_griddata2(data):
            #grid_data = np.ma.masked_where(np.isnan(data), data)
            grid_data = data
            return grid_data
                    
    # em fluxes
    if field == 'femag' or field=='femag_norm':
        grid_data = my_griddata2(datdict['fe_mag']*datdict['gdet'])
        #grid_data = np.sign(grid_data)

    elif field == 'femag_prims' or field=='femag_prims_norm':
        grid_data = my_griddata2(datdict['fe_mag_prims']*datdict['gdet'])
        #grid_data = np.sign(grid_data)
        
    elif field == 'femag_starf' or field=='femag_starf_norm':
        grid_data = my_griddata2(datdict['fe_mag_starf']*datdict['gdet'])
        #grid_data = np.sign(grid_data)

    # matter fluxes
    elif field == 'fehd' or field=='fehd_norm':
        grid_data = my_griddata2(datdict['fe_hd']*datdict['gdet'])
 
       
    # magnetic field
    elif field == 'bflux':
        grid_data = np.abs(my_griddata2(datdict['Bcon'][1]*datdict['gdet']))
    elif field == 'bpol':
        Bcon1 = datdict['Bcon'][1]
        Bcon2 = datdict['Bcon'][2]
        Bcon3 = datdict['Bcon'][3]
        if METRIC_OUT=='KS':
            (_, Bcov1, Bcov2, Bcov3) = lowerKS(0*Bcon1,Bcon1,Bcon2,Bcon3,spin,r,th)
        elif METRIC_OUT=='BL':
            (_, Bcov1, Bcov2, Bcov3) = lowerBL(0*Bcon1,Bcon1,Bcon2,Bcon3,spin,r,th)
        else:
            raise Exception()        
        Bpol = Bcon1*Bcov1 + Bcon2*Bcov2
        grid_data = np.abs(my_griddata2(Bpol))
        grid_data = np.log10(np.sqrt(grid_data))
        grid_data = np.ma.masked_where(np.isnan(grid_data), grid_data)        
    elif field == 'bratio':
        grid_Bph = my_griddata2(datdict['Bcon'][3])  
        grid_Br = my_griddata2(datdict['Bcon'][1])
        grid_data = grid_Bph/grid_Br    
        #if np.abs(spin)>0:
        #    grid_data *= np.sign(spin)    
    elif field == 'bratio_sign':
        grid_Bph = my_griddata2(datdict['Bcon'][3])  
        grid_Br = my_griddata2(datdict['Bcon'][1])
        #grid_data = np.sign(grid_Bph)/np.sign(grid_Br)    
        grid_data = np.sign(grid_Bph/(grid_Br+1.e-6))    
        #if np.abs(spin)>0:
        #    grid_data *= np.sign(spin)    
    elif field == 'omegafield_th':
        grid_data = my_griddata2(datdict['omega_th'])
        #grid_data = np.sign(grid_data)
    elif field == 'omegafield_r':
        grid_data = my_griddata2(datdict['omega_r'])
        #grid_data = np.sign(grid_data)
    elif field == 'omegafluid':
        grid_data = my_griddata2(datdict['ucon'][3]/datdict['ucon'][0])
    elif field == 'sigma2':
        grid_data = my_griddata2(datdict['bsq'])/my_griddata2(datdict['rho'])
    elif field == 'beta2':
        grid_data = my_griddata2((datdict['gamma_adiab']-1)*datdict['uint'])/my_griddata2(datdict['bsq'])
    elif field == 'logsigma2':
        grid_data = np.log10(my_griddata2(datdict['bsq'])/my_griddata2(datdict['rho']))
    elif field == 'logbeta2':
        grid_data = np.log10(my_griddata2((datdict['gamma_adiab']-1)*datdict['uint'])/my_griddata2(datdict['bsq']))
    else:
        grid_data = my_griddata2(datdict[field])

                 
    if verbose: print(field,'min/max',np.min(grid_data),np.max(grid_data))           
    return (grid_x,grid_z,grid_data)

