# Andrew Chael, March 2024
# IN PROGRESS - MODIFY POST-LOADING AND ADD TO OBJECT

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

class simdata2D(object):
    def __init__(self, filename, metric, r, th):
        self.filename = filename
        if metric not in ['KS','BL']:
            raise Exception("metric must be 'KS' or 'BL'")
        self.metric = metric
        
        # coordinates
        if r.shape != th.shape:
            raise Exception("grid shapes are inconsistent!")        
        self.r = r
        self.th = th
            
        self.n1 = r.shape[0]
        self.n2 = th.shape[1]

        self.data = {}       
        self.data_derived = {}
        
    def setdata(self, field, array):
        if array.shape != (self.n1,self.n2):
            raise Exception(field, " shape is not consistent with grid ", self.n1, self.n2)

        self.data[field] = array
   
    def set_derived_quantities(self):
        """compute derived data, i.e. data not averaged directly"""
        # conv_vel
        if self.metric=='KS':
            conv_vel = conv_vel_KS
        elif self.metric=='BL':  
            conv_vel = conv_vel_BL
                    
        # metric determinant
        if self.metric=='KS':
            self.gdet = gdetKS(self.spin, self.r, self.th) 
        elif self.metric=='BL':  
            self.gdet = gdetBL(self.spin, self.r, self.th)  

        # Lorentz Factor
        if self.metric=='KS':
            gcon00 = gconKS(self.spin,self.r,self.th)[0][0]
        elif self.metric=='BL':  
            gcon00 = gconBL(self.spin,self.r,self.th)[0][0] 
        self.data_derived['lorentz'] = self.data['u0']/np.sqrt(-gcon00)

        # A_phi (poloidal field lines)
        dth = 0.5*(self.th[:,2:] - self.th[:,:-2]) # approx dtheta        
        integrand = (self.data['B1'][:,1:-1]*self.gdet[:,1:-1])*dth
        aphi_th = np.zeros(self.r.shape)
        for i in range(self.n2-2):
            if i == 0: 
                aphi_th[:,0] = (integrand*dth)[:,0]
                aphi_th[:,0+1] = (integrand*dth)[:,0]
            else: 
                aphi_th[:,i+1] = aphi_th[:,i] + (integrand*dth)[:,i]
        aphi_th[:,-1] = aphi_th[:,-2]
        self.data_derived['Aphi'] = aphi_th
        
        # fieldline angular speed
        self.data_derived['Omega_F'] = self.data['sF13']/self.data['B1']
        #self.data_derived['Omega_F'] = self.data['sF23']/self.data['B2']
        
        # fluid angular speed
        # TODO - average directly?
        self.data_derived['Omega'] = self.data['u3']/self.data['u0']
        #(u0avg,u1avg,u2avg,u3avg) = conv_vel(self.data['U1'], self.data['U2'], self.data['U3'], self.spin, self.r, self.th)
        #self.data_derived['Omega'] = u3avg/u0avg
                          
        # energy fluxes
        if self.metric=='KS':
            g = gcovKS(self.spin,self.r,self.th)
        elif self.metric=='BL':
            g = gcovBL(self.spin,self.r,self.th)           
        fe_mag = -1*(g[0][0]*self.data['T01_mag'] + g[0][1]*self.data['T11_mag'] + g[0][2]*self.data['T12_mag'] + g[0][3]*self.data['T13_mag'])
        fe_hd  = -1*(g[0][0]*self.data['T01_hd']  + g[0][1]*self.data['T11_hd']  + g[0][2]*self.data['T12_hd']  + g[0][3]*self.data['T13_hd'])
        self.data_derived['fe_mag'] = fe_mag
        self.data_derived['fe_hd'] = fe_hd     
        if 'T01_rad' in self.data.keys():
            fe_rad = -1*(g[0][0]*self.data['T01_rad'] + g[0][1]*self.data['T11_rad'] + g[0][2]*self.data['T12_rad'] + g[0][3]*self.data['T13_rad'])
            self.data_derived['fe_rad'] = fe_rad       
                
        # angular momentum fluxes
        fj_mag = (g[3][0]*self.data['T01_mag'] + g[3][1]*self.data['T11_mag'] + g[3][2]*self.data['T12_mag'] + g[3][3]*self.data['T13_mag'])
        fj_hd  = (g[3][0]*self.data['T01_hd']  + g[3][1]*self.data['T11_hd']  + g[3][2]*self.data['T12_hd']  + g[3][3]*self.data['T13_hd'])
        self.data_derived['fj_mag'] = fj_mag
        self.data_derived['fj_hd'] = fj_hd
        if 'T01_rad' in self.data.keys():
            fj_rad = (g[3][0]*self.data['T01_rad'] + g[3][1]*self.data['T11_rad'] + g[3][2]*self.data['T12_rad'] + g[3][3]*self.data['T13_rad'])
            self.data_derived['fj_rad'] = fj_rad       
        
        # Temperature Ratio
        if 'te' in self.data.keys():
            self.data_derived['tratio'] = self.data['ti']/self.data['te']
            
        return
        
        
def read_koral_hdf52D(filein, verbose=True, compute_derived=True):
    """read phi-averaged or sliced hdf5 file"""
    
    if verbose: print('reading hdf5 ', filein, '....')
    
    # load data     
    fin = h5py.File(filein,'r')

    # get info from the header
    metric_run = fin['header']['metric_run'][()]
    metric_out = fin['header']['metric_out'][()]
    if not isinstance(metric_run, str): metric_run = metric_run.decode('utf-8')
    if not isinstance(metric_out, str): metric_out = metric_out.decode('utf-8')

    # coords  
    r = fin['grid_out']['r'][:]
    th = fin['grid_out']['th'][:]
    
    if r.shape != th.shape:
        raise Exception("grid shapes are inconsistent!")
    if len(r.shape)!=2:
        raise Exception("grid must be 2D, but len(r.shape)=",len(r.shape))
    if r.shape[0] != fin['header']['n1'][()]: 
        raise Exception("grid shape n1 inconsistent in ",filein)
    if r.shape[1] != fin['header']['n2'][()]: 
        raise Exception("grid shape n2 inconsistent in ",filein)
                
    # output object
    outdata = simdata2D(filein, metric_out, r, th)

    # header quantities
    outdata.spin = fin['header']['bhspin'][()].astype('f')
    outdata.gamma_adiab = fin['header']['gam'][()]
    
       
    # get all hdf5 quantites 
    for key in fin['quants'].keys():
        outdata.setdata(key, fin['quants'][key][:])
                            
    # close hdf5 file
    fin.close()
    
    # compute derived quantities
    if compute_derived:
        outdata.set_derived_quantities()
    
    return outdata

# grid the poloidal profiles
def gridprofile(datobj, field, xlim=40, zlim1=-20, zlim2=20, ngrid=500j, verbose=True, reggrid=True):
  
    spin = datobj.spin
    horiz = 1 + np.sqrt(1-spin**2)    
    omegaH = spin / (2*horiz)
    if spin==0: 
        omegaISCO = 1/(6**1.5)
        omegaH = omegaISCO

    metric = datobj.metric
    r = datobj.r
    th = datobj.th
    gdet = datobj.gdet
    
    datdict = datobj.data
    datdict.update(datobj.data_derived)
    
    # put on a regular grid or keep simulation grid? 
    if reggrid:
        points=((r*np.sin(th)).flatten(), (r*np.cos(th)).flatten())    
        grid_z, grid_x = np.mgrid[zlim1:zlim2:ngrid,  0:xlim:ngrid]  
        grid_r = np.sqrt(grid_z**2 + grid_x**2) 
        def my_griddata(data):
            grid_data = griddata(points, data.flatten(), (grid_x, grid_z), method='cubic', fill_value=np.nan) 
            grid_data = np.ma.masked_where(np.isnan(grid_data), grid_data)
            return grid_data
    else:
        grid_x = (r*np.sin(th))
        grid_z = (r*np.cos(th))  
        grid_r = np.sqrt(grid_z**2 + grid_x**2) 
        def my_griddata(data):
            grid_data = data
            return grid_data
    
    # log or not
    logfield=False
    if field[0:3]=='log':
        logfield=True
        field = field[3:]
                  
    # em fluxes
    if field == 'femag' or field=='femag_norm':
        grid_data = my_griddata(datdict['fe_mag']*gdet)

    # matter fluxes
    elif field == 'fehd' or field=='fehd_norm':
        grid_data = my_griddata(datdict['fe_hd']*gdet)
 
    # rad fluxes
    elif field == 'ferad' or field=='ferad_norm':
        grid_data = my_griddata(datdict['fe_rad']*gdet)
              
    # magnetic field
    elif field == 'bflux':
        grid_data = np.abs(my_griddata(datdict['B1']*gdet))
    elif field == 'bpol':
        Bcon1 = datdict['B1']
        Bcon2 = datdict['B2']
        Bcon3 = datdict['B3']
        if metric=='KS':
            (_, Bcov1, Bcov2, Bcov3) = lowerKS(0*Bcon1,Bcon1,Bcon2,Bcon3,spin,r,th)
        elif metric=='BL':
            (_, Bcov1, Bcov2, Bcov3) = lowerBL(0*Bcon1,Bcon1,Bcon2,Bcon3,spin,r,th)
        else:
            raise Exception()        
        Bpol = Bcon1*Bcov1 + Bcon2*Bcov2
        grid_data = np.sqrt(np.abs(my_griddata(Bpol)))


    elif field == 'bratio':
        grid_Bph = my_griddata(datdict['B3'])  
        grid_Br = my_griddata(datdict['B1'])
        grid_data = grid_Bph/grid_Br    
    elif field == 'bratio_sign':
        grid_Bph = my_griddata(datdict['B3'])  
        grid_Br =  my_griddata(datdict['B1'])
        grid_data = np.sign(grid_Bph*grid_Br)    
        #grid_data = np.sign(grid_Bph/(grid_Br+1.e-6))            
    elif field == 'omegafieldnorm':
        grid_data = my_griddata(datdict['Omega_F'])/omegaH
    elif field == 'omegafield':
        grid_data = my_griddata(datdict['Omega_F'])
    elif field == 'omegafluid':
        grid_data = my_griddata(datdict['Omega'])
    elif field == 'sigma2':
        grid_data = my_griddata(datdict['bsq'])/my_griddata(datdict['rho'])
    elif field == 'beta2':
        grid_data = my_griddata(datdict['pgas'])/my_griddata(datdict['bsq'])
    elif field == 'temp':
        grid_data = my_griddata(datdict['Tgas'])
    elif field == 'te2':
        grid_data = my_griddata(datdict['pe'])/my_griddata(datdict['rho']) * (MU_E*MRATIO) / TEFAC
    elif field == 'ti2':
        grid_data = my_griddata(datdict['pi'])/my_griddata(datdict['rho']) * (MU_I) / TPFAC
    else:
        grid_data = my_griddata(datdict[field])

    if logfield:
        grid_data = np.log10(grid_data)
        field = 'log'+field

    grid_data = np.ma.masked_where(np.isnan(grid_data), grid_data)                         
    if verbose: print(field,'min/max',np.min(grid_data),np.max(grid_data))           
    return (grid_x,grid_z,grid_data)   

