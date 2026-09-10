# Andrew Chael, March 2024
# IN PROGRESS - MODIFY POST-LOADING AND ADD TO OBJECT

import pylab
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from collections import OrderedDict
from scipy.interpolate import griddata
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from metricKS import *
from koralopacities import *
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

# upper-triangle stress tensor components we store
TIDX = [(0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]

METRIC_DEFAULT = 'KS'
KLEINNISHINA = False # we had KN turned off in KORAL runs for some reason...


def _tostr(val):
    """hdf5 string datasets come back as bytes on py3"""
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return str(val)


class simdata3D(object):
    """A raw KORAL ipole-format hdf5 dump plus the quantities derived from it.

    The derived quantities are exposed lazily through field_names()/field() so a
    caller can compute and consume them one at a time instead of materializing
    every 3D array at once.  That is what lets the 3D time-average stream over
    files without holding more than one running sum in memory.
    """

    def __init__(self, filename, metric=METRIC_DEFAULT, kn=KLEINNISHINA, verbose=False):
        if metric not in ('KS','BL'):
            raise Exception("metric must be 'KS' or 'BL'")
        self.filename = filename
        self.metric = metric
        self.kn = kn
        self.verbose = verbose

        self._read()

    @property
    def shape(self):
        return self.rho.shape

    def _read(self):
        """Read the header, the grid and the primitive variables out of the dump"""
        if self.verbose: print('reading hdf5 ', self.filename, '....')

        with h5py.File(self.filename, 'r') as fin:
            self.time = float(fin['t'][()])

            # get info from the header
            head = fin['header']
            self.spin = head['bhspin'][()].astype('f')
            self.horiz = 1 + np.sqrt(1 - self.spin**2)
            self.metric_run = _tostr(head['metric_run'][()])
            self.metric_out = _tostr(head['metric_out'][()])
            self.has_radiation = bool(head['has_radiation'][()])
            self.has_electrons = bool(head['has_electrons'][()])

            self.file_number = head['file_number'][()]
            self.problem_number = head['problem_number'][()]
            self.version = head['version'][()]

            self.masssolar = head['units']['M_bh'][()]
            self.mass_cgs = self.masssolar * MSUN_CGS
            self.l_unit = head['units']['L_unit'][()] # length code2cgs = GM/c^2
            self.u_unit = head['units']['U_unit'][()] # energy density code2cgs = c^8/G^3M^2
            self.m_unit = head['units']['M_unit'][()] # mass density code2cgs = c^6/G^3M^2

            if self.verbose:
                print("M:%s, L:%s, U:%s, rho:%s"
                      % (self.mass_cgs, self.l_unit, self.u_unit, self.m_unit))

            if self.metric_out not in ('KS','BL'):
                raise Exception("metric_out of KORAL h5 file must be KS or BL, got '%s'"
                                % self.metric_out)

            self.n1 = int(head['n1'][()])
            self.n2 = int(head['n2'][()])
            self.n3 = int(head['n3'][()])
            self.gamma_head = head['gam'][()]

            # get coordinates in OUTCOORDS (should be KS)
            self.r  = fin['grid_out']['r'][:,:,:]
            self.th = fin['grid_out']['th'][:,:,:]
            self.ph = fin['grid_out']['ph'][:,:,:]

            # get simulation primitive variables
            quants = fin['quants']
            self.rho = quants['rho'][:,:,:]
            self.uint = quants['uint'][:,:,:]

            self.u1_velr = quants['U1'][:,:,:]
            self.u2_velr = quants['U2'][:,:,:]
            self.u3_velr = quants['U3'][:,:,:]

            self.B1 = quants['B1'][:,:,:]
            self.B2 = quants['B2'][:,:,:]
            self.B3 = quants['B3'][:,:,:]

            if self.has_radiation:
                self.erad = quants['erad'][:,:,:]
                self.has_photons = 'nphot' in quants
                if self.has_photons:
                    self.nphot = quants['nphot'][:,:,:]
                else:
                    self.nphot = np.zeros(self.erad.shape)

                self.ur1_velr = quants['F1'][:,:,:]
                self.ur2_velr = quants['F2'][:,:,:]
                self.ur3_velr = quants['F3'][:,:,:]
            else:
                self.has_photons = False

            if self.has_electrons:
                self.TeK = quants['te'][:,:,:]
                self.TiK = quants['ti'][:,:,:]
                self.gamma_adiab = quants['gammagas'][:,:,:]
            else:
                self.gamma_adiab = self.gamma_head

    def set_derived_quantities(self):
        """Metric, 4-velocities, b^mu, and the plasma parameters"""
        spin, r, th = self.spin, self.r, self.th

        # Metric
        if self.metric == 'KS':
            self.gcon = gconKS(spin, r, th)
            lower = lowerKS
            conv_vel = conv_vel_KS
            trans_cov = trans_cov_bl2ks
            invconv_vel = invconv_vel_KS
        else:
            self.gcon = gconBL(spin, r, th)
            lower = lowerBL
            conv_vel = conv_vel_BL
            trans_cov = trans_cov_ks2bl
            invconv_vel = invconv_vel_BL

        # get 4-velocity
        (u0,u1,u2,u3) = conv_vel(self.u1_velr, self.u2_velr, self.u3_velr, spin, r, th)
        (u0_l,u1_l,u2_l,u3_l) = lower(u0, u1, u2, u3, spin, r, th)

        if self.has_radiation:
            (ur0,ur1,ur2,ur3) = conv_vel(self.ur1_velr, self.ur2_velr, self.ur3_velr, spin, r, th)
            (ur0_l,ur1_l,ur2_l,ur3_l) = lower(ur0, ur1, ur2, ur3, spin, r, th)

        # get magnetic field 4-vector and b^2
        B1, B2, B3 = self.B1, self.B2, self.B3
        b0 = u1_l*B1 + u2_l*B2 + u3_l*B3
        b1 = (B1 + b0*u1)/u0
        b2 = (B2 + b0*u2)/u0
        b3 = (B3 + b0*u3)/u0

        (b0_l,b1_l,b2_l,b3_l) = lower(b0, b1, b2, b3, spin, r, th)
        bsq = b0*b0_l + b1*b1_l + b2*b2_l + b3*b3_l

        # change coordinates
        if self.metric_out != self.metric:
            (u0,u1,u2,u3) = trans_cov(u0, u1, u2, u3, spin, r, th)
            (u0_l,u1_l,u2_l,u3_l) = lower(u0, u1, u2, u3, spin, r, th)
            self.u1_velr, self.u2_velr, self.u3_velr = invconv_vel(u0, u1, u2, u3, spin, r, th)

            (b0,b1,b2,b3) = trans_cov(b0, b1, b2, b3, spin, r, th)
            (b0_l,b1_l,b2_l,b3_l) = lower(b0, b1, b2, b3, spin, r, th)

            if self.has_radiation:
                (ur0,ur1,ur2,ur3) = trans_cov(ur0, ur1, ur2, ur3, spin, r, th)
                (ur0_l,ur1_l,ur2_l,ur3_l) = lower(ur0, ur1, ur2, ur3, spin, r, th)
                self.ur1_velr, self.ur2_velr, self.ur3_velr = invconv_vel(ur0, ur1, ur2, ur3, spin, r, th)

        # (re) derive B-field 3 vector in case coordinates changed
        self.B1 = b1*u0 - b0*u1
        self.B2 = b2*u0 - b0*u2
        self.B3 = b3*u0 - b0*u3

        self.ucon = (u0,u1,u2,u3)
        self.ucov = (u0_l,u1_l,u2_l,u3_l)
        self.bcon = (b0,b1,b2,b3)
        self.bcov = (b0_l,b1_l,b2_l,b3_l)
        self.bsq = bsq
        if self.has_radiation:
            self.urcon = (ur0,ur1,ur2,ur3)
            self.urcov = (ur0_l,ur1_l,ur2_l,ur3_l)

        # plasma parameters
        self.pgas = (self.gamma_adiab - 1.)*self.uint           # pressure
        self.w = (self.rho + self.pgas + self.uint)             # enthalpy
        self.TgasK = (self.pgas / self.rho) * (MU_GAS / TPFAC)  # gas temperature in Kelvin

        # radiation derived quantities
        if self.has_radiation:
            self._set_radiation_quantities()

        # electron derived quantities
        if self.has_electrons:
            self._set_electron_quantities()

        # the table of output quantities depends on has_radiation/has_electrons
        self._build_fields()

        return

    def _set_electron_quantities(self):
        """electron/ion pressures and the heating functions"""
        self.pi = (self.rho/MU_I)*TPFAC*self.TiK  # ion pressure, code units
        self.pe = (self.rho/MU_E)*TPFAC*self.TeK  # electron pressure, code units

        # heating functions (unitless)
        self.deltaeK = deltaeKawazura(2*self.pi/self.bsq, self.TeK/self.TiK)
        self.deltaeZ = deltaeZhdankin(TEFAC*self.TeK, TPFAC*self.TiK/MU_I, MRATIO)

        return

    def _set_radiation_quantities(self):
        """fluid frame radiation quantities and the opacities, all in code units"""
        erad = self.erad

        # fluid frame radiation energy density
        erad_hat = np.zeros(erad.shape)
        for i in range(4):
            for j in range(4):
                erad_hat += ((4./3.)*erad*self.urcon[i]*self.urcon[j]
                             + (1./3.)*erad*self.gcon[i][j])*self.ucov[i]*self.ucov[j]
        self.erad_hat = erad_hat

        # fluid frame photon number
        if self.has_photons:
            nphot_hat = np.zeros(self.nphot.shape)
            for i in range(4):
                nphot_hat += -self.nphot*self.urcon[i]*self.ucov[i]
            self.nphot_hat = nphot_hat

            # radiation temperature in the fluid frame
            self.TradK = erad_hat/(2.7012*nphot_hat) * (self.mass_cgs*C_CGS*C_CGS / KBOLTZ_CGS) # in Kelvin TODO check
        else:
            self.TradK = ((erad_hat*self.u_unit / A_RAD_CGS))**(0.25) # in Kelvin
            self.nphot_hat = ((A_RAD_CGS * (self.TradK**3) / (2.70118*KBOLTZ_CGS))
                              * (self.mass_cgs*C_CGS*C_CGS / self.u_unit))

        # opacities
        # u_unit = c^8/G^3 M^2 is the transformation for energy density from code --> cgs
        rhocgs = self.rho * (self.u_unit / (C_CGS*C_CGS))
        Bmagcgs = np.sqrt(self.bsq) * np.sqrt(4*np.pi*self.u_unit)

        if not self.has_electrons:
            self.TeK = self.TgasK

        # opacities (compute in cgs from koralopacities.py and return to code units)
        # TODO include scattering?
        (synemisopac, synabsorbopac) = synopac(rhocgs, self.TeK, self.TradK, Bmagcgs)
        (ffemisopac, ffabsorbopac) = ffopac(rhocgs, self.TeK, self.TradK)
        comptabsorbopac = comptopac(rhocgs, self.TeK, self.TradK, kn=self.kn)

        # return to code units
        self.synemisopac = synemisopac * self.l_unit
        self.synabsorbopac = synabsorbopac * self.l_unit
        self.ffemisopac = ffemisopac * self.l_unit
        self.ffabsorbopac = ffabsorbopac * self.l_unit
        self.comptabsorbopac = comptabsorbopac * self.l_unit

        # blackbody energy density (code units)
        self.BBenergy = A_RAD_CGS*(self.TeK**4) / self.u_unit

        return

    ###########################################################################
    # table of output quantities
    ###########################################################################
    def _build_fields(self):
        """Register every output quantity as a name -> zero-argument callable.

        Nothing is evaluated here, so a caller can pull the quantities out one at
        a time rather than holding all of them on the 3D grid at once.
        """
        f = OrderedDict()

        # Primitive Quantities
        f['rho']  = lambda: self.rho
        f['uint'] = lambda: self.uint

        f['B1'] = lambda: self.B1
        f['B2'] = lambda: self.B2
        f['B3'] = lambda: self.B3

        f['U1'] = lambda: self.u1_velr
        f['U2'] = lambda: self.u2_velr
        f['U3'] = lambda: self.u3_velr

        # Derived Quantities
        # we don't need the lorentz factor for axisymmetric metric, we can get it from u^0
        f['pgas']    = lambda: self.pgas # redundant for MHD but useful with variable adiabatic index
        f['bsq']     = lambda: self.bsq
        f['sigma']   = lambda: self.bsq / self.rho
        f['sigmaw']  = lambda: self.bsq / self.w
        f['Tgas']    = lambda: self.TgasK # TODO in kelvin or not?
        f['beta']    = lambda: self.pgas / (0.5*self.bsq)
        f['betainv'] = lambda: (0.5*self.bsq) / self.pgas

        f['absB1'] = lambda: np.abs(self.B1)
        f['absB2'] = lambda: np.abs(self.B2)
        f['absB3'] = lambda: np.abs(self.B3)

        for i in range(4):
            f['b%d' % i] = (lambda i=i: self.bcon[i])
        for i in range(4):
            f['u%d' % i] = (lambda i=i: self.ucon[i])

        # spatial components of Maxwell (contravariant)
        f['sF12'] = lambda: self.bcon[1]*self.ucon[2] - self.bcon[2]*self.ucon[1]
        f['sF13'] = lambda: self.bcon[1]*self.ucon[3] - self.bcon[3]*self.ucon[1]
        f['sF23'] = lambda: self.bcon[2]*self.ucon[3] - self.bcon[3]*self.ucon[2]

        # Tmunu mag (contravariant)
        for (i,j) in TIDX:
            f['T%d%d_mag' % (i,j)] = (lambda i=i, j=j:
                self.bsq*self.ucon[i]*self.ucon[j] - self.bcon[i]*self.bcon[j]
                + 0.5*self.bsq*self.gcon[i][j])

        # Tmunu mat (contravariant)
        for (i,j) in TIDX:
            f['T%d%d_hd' % (i,j)] = (lambda i=i, j=j:
                self.w*self.ucon[i]*self.ucon[j] + self.pgas*self.gcon[i][j])

        # rho-weighted quantities
        f['rhosq']   = lambda: self.rho*self.rho
        f['rhobsq']  = lambda: self.rho*self.bsq
        f['rhouint'] = lambda: self.rho*self.uint
        f['rhopgas'] = lambda: self.rho*self.pgas
        f['rhoscaleheight'] = lambda: self.rho*np.abs(self.th - 0.5*np.pi)

        for i in range(4):
            f['rhou%d' % i] = (lambda i=i: self.rho*self.ucon[i])

        f['rhoabsB1'] = lambda: self.rho*np.abs(self.B1)
        f['rhoabsB2'] = lambda: self.rho*np.abs(self.B2)
        f['rhoabsB3'] = lambda: self.rho*np.abs(self.B3)

        f['rhoB1'] = lambda: self.rho*self.B1
        f['rhoB2'] = lambda: self.rho*self.B2
        f['rhoB3'] = lambda: self.rho*self.B3

        for i in range(4):
            f['rhob%d' % i] = (lambda i=i: self.rho*self.bcon[i])

        if self.has_radiation:
            f['erad']  = lambda: self.erad  # radiation frame
            f['nphot'] = lambda: self.nphot # radiation frame

            f['F1'] = lambda: self.ur1_velr
            f['F2'] = lambda: self.ur2_velr
            f['F3'] = lambda: self.ur3_velr

            for i in range(4):
                f['ur%d' % i] = (lambda i=i: self.urcon[i])

            for (i,j) in TIDX:
                f['T%d%d_rad' % (i,j)] = (lambda i=i, j=j:
                    (4./3.)*self.erad*self.urcon[i]*self.urcon[j]
                    + (1./3.)*self.erad*self.gcon[i][j])

            # fluid frame quantities
            f['erad_hat']  = lambda: self.erad_hat
            f['nphot_hat'] = lambda: self.nphot_hat
            f['Trad']      = lambda: self.TradK

            # opacities
            f['opac_syn'] = lambda: self.synabsorbopac
            f['opac_ff']  = lambda: self.ffabsorbopac
            # TODO don't double count opac_compt with emis_compt
            # f['opac_compt'] = lambda: self.comptabsorbopac

            f['emis_syn']   = lambda: -self.BBenergy*self.synemisopac
            f['emis_ff']    = lambda: -self.BBenergy*self.ffemisopac
            f['emis_compt'] = lambda: self.erad_hat*self.comptabsorbopac # should be negative

            # rho weighted quantites (?)
            # TODO do we want ebar weighted quantities?
            f['rhoehat'] = lambda: self.rho*self.erad_hat
            f['rhonhat'] = lambda: self.rho*self.nphot_hat
            f['rhoTrad'] = lambda: self.rho*self.TradK

            for i in range(4):
                f['rhour%d' % i] = (lambda i=i: self.rho*self.urcon[i])

        if self.has_electrons:
            f['ti'] = lambda: self.TiK # TODO in kelvin or not?
            f['te'] = lambda: self.TeK # TODO in kelvin or not?
            f['gammagas'] = lambda: self.gamma_adiab

            f['pi'] = lambda: self.pi
            f['pe'] = lambda: self.pe

            # derived quantities
            f['deltaeK'] = lambda: self.deltaeK
            f['deltaeZ'] = lambda: self.deltaeZ

            # rho weighted quantites (?)
            f['rhope'] = lambda: self.rho*self.pe
            f['rhopi'] = lambda: self.rho*self.pi
            f['rhogammagas'] = lambda: self.rho*self.gamma_adiab

        self._fields = f

        return

    def field_names(self, fields=None, exclude=None):
        """Names of the available output quantities, optionally filtered"""
        names = list(self._fields.keys())
        if fields:
            missing = [k for k in fields if k not in self._fields]
            if missing:
                raise Exception("unknown field(s) %s in %s -- available: %s"
                                % (missing, self.filename, ', '.join(names)))
            names = [k for k in names if k in set(fields)]
        if exclude:
            names = [k for k in names if k not in set(exclude)]
        return names

    def field(self, name):
        """Evaluate one output quantity on the full 3D grid"""
        arr = self._fields[name]()
        # gammagas is a scalar for runs without electrons, broadcast for uniformity
        if np.ndim(arr) == 0:
            arr = np.full(self.rho.shape, arr)
        return arr

    def write_header(self, fout, n3, ndim):
        """Write the output /header group, copying units and geom from the dump.

        fout  -- an open, writable h5py.File for the output file
        n3    -- size of the phi axis of the output grid: 1 for a phi-reduced
                 file, self.n3 for one that keeps the phi axis
        ndim  -- dimensionality of the output grid: 2 or 3, to match n3
        """
        grp = fout.create_group('header')
        grp.create_dataset('bhspin', data=self.spin)
        grp.create_dataset('file_number', data=self.file_number)
        # header gam is the scalar initial value; the per-cell value is quants/gammagas
        grp.create_dataset('gam', data=self.gamma_head)
        grp.create_dataset('has_electrons', data=int(self.has_electrons))
        grp.create_dataset('has_radiation', data=int(self.has_radiation))
        grp.create_dataset('metric_out', data=self.metric)
        grp.create_dataset('metric_run', data=self.metric_run)
        grp.create_dataset('n1', data=self.n1)
        grp.create_dataset('n2', data=self.n2)
        grp.create_dataset('n3', data=n3)
        grp.create_dataset('ndim', data=ndim)
        grp.create_dataset('problem_number', data=self.problem_number)
        grp.create_dataset('version', data=self.version)

        with h5py.File(self.filename, 'r') as fsrc:
            grp.copy(fsrc['header']['units'], grp)
            grp.copy(fsrc['header']['geom'], grp)

        return


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

