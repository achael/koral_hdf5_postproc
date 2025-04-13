# KORAL free-free, synchrotron, electron scattering, and compton cooling
# ALL IN CGS! 

import numpy as np

# pure hydrogen is hard coded
MU_I = 1
MU_E = 1
MU_GAS = 0.5
KLEINNISHINA=False # we had KN turned off in KORAL runs for some reason...

# constants
MP_CGS = 1.67262158e-24
ME_CGS = 9.1094e-28
MRATIO = MP_CGS / ME_CGS
KBOLTZ_CGS = 1.3806488e-16
SIGMA_RAD_CGS = 5.670367e-5
C_CGS = 2.9979246e10
TPFAC = KBOLTZ_CGS/(MP_CGS*C_CGS*C_CGS) # kelvin to dimensionless units for Tp
TEFAC = KBOLTZ_CGS/(ME_CGS*C_CGS*C_CGS) # kelvin to dimensionless units for Te


# all opacities rho*kappa returned in cgs units of 1/cm 

# synchrotron
def synopac(rhocgs, TeK, TradK, Bmagcgs):

    # all quantities are in cgs! 
    ne = rhocgs/(MU_E*MP_CGS)
    
    # emission
    emis = 1.59e-30*ne*Bmagcgs*Bmagcgs/(TeK*TeK) #kappa_e * rho
    
    # absorption
    zetaB = 1.74498e23 * TradK/(TeK*TeK)   
    zetadenom = 1.79*np.cbrt(zetaB**5 * Bmagcgs**4) + 1.35*np.cbrt(zetaB**7 * Bmagcgs**2) + 0.248*(zetaB**3)
    IaByBrad = Bmagcgs*Bmagcgs/zetadenom
    absorp = 2.13e39*ne*IaByBrad / (TeK**5)  #kappa_a * rho
    
    # suppress at nonrelativistic temperatures
    thetae = TeK*1.6863e-10
    Terelfactor = thetae**2 / (1 + thetae**2)
    emis *= Terelfactor
    absorp *= Terelfactor
    
    return (emis, absorp) # units of 1/cm
    
# free-free
def ffopac(rhocgs, TeK, TradK):
    # all quantities in cgs! 
    ne = rhocgs/(MU_E*MP_CGS)
    nibar = rhocgs/(MP_CGS)  # this is (X+Y) rho/mp, assuming X+Y=0, metallicity 0 
    
    # emission
    emis = (6.2e-24*1.2*ne*nibar*np.sqrt(TeK)/(TeK**4))   
    emis *= (1 + 4.4e-10*TeK) # relativistic correction

    # absorption
    xi = TradK/TeK   
    absorp = emis*1.04656*np.log1p(1.6*xi) / (xi**3)
    
    return (emis, absorp) # units of 1/cm

# scattering    
def scatopac(rhocgs, TeK, TradK, kn=False):

    KNfac=1
    
    # we had KN turned off in KORAL 
    if kn:
        Tkn = np.sqrt(TeK*TeK + TradK*TradK)
        KNfac = 1./(1. + (Tkn/4.5e8)**0.86) # TODO is this even right? 
    
    # all quantities in cgs
    absorp = 0.2 * (2./MU_E) * rhocgs * KNfac
    
    return absorp # units of 1/cm

# Comptonization    
def comptopac(rhocgs, TeK, TradK, kn=False):
    # Compton G0 / Crad
    kappaes = scatopac(rhocgs, TeK, TradK, kn=kn)
    
    thetae = TeK*TEFAC
    thetarad = TradK*TEFAC
    
    #comptG0 = kappaes*Erad*(4 * (thetarad-thetae))*(1 + 3.683*thetae + 4*thetae*thetae) / (1 + 4*thetae)   
    comptopac = kappaes*(4 * (thetarad-thetae))*(1 + 3.683*thetae + 4*thetae*thetae) / (1 + 4*thetae)
    
    return comptopac # units of 1/cm
    
# Kawazura heating function
def deltaeKawazura(betai, Tratio):

    uioue = 35./(1 + ((betai/15.)**(-1.4))*np.exp(-0.1*Tratio))
    deltae = 1./(1. + uioue)
    return deltae
    
def deltaeZhdankin(thetae, thetai, mratio):

    
    gmeani = calc_meanlorentz(thetai)
    gmeane = calc_meanlorentz(thetae)
    gyroratio = mratio *np.sqrt((gmeani*gmeani-1)/(gmeane*gmeane-1))
    uioue = gyroratio**(2./3.)
    deltae = 1./(1. + uioue)
    return deltae

def calc_meanlorentz(theta):
    return 1 + 3*theta*(0.5 + 1.5*theta)/(1 + 1.5*theta)
    
