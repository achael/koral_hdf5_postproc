#Kerr-Schild and Boyer-Lindquist metric functions

import numpy as np

##################################
# Kerr-Schild
##################################

# covariant metric in KS
def gcovKS(spin,r,th):

    a = spin
    a2 = a**2
    r2 = r**2
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
        
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2
    Pi = (r2+a2)**2 - a2*Delta*sth2
    
    if isinstance(r,np.ndarray):
        gcov = np.zeros(np.hstack(((4,4),r.shape)))
    else:
        gcov = np.zeros((4,4))
        
    gcov[0][0] = -(1-2*r/Sigma)
    gcov[1][1] = (1 + 2*r/Sigma)
    gcov[2][2] = Sigma
    gcov[3][3] = sth2 *(Sigma + (1+2*r/Sigma)*sth2*a2) 
    
    gcov[0][3] = -2*r*a*sth2 / Sigma
    gcov[3][0] = gcov[0][3]
        
    gcov[0][1] = 2*r/Sigma
    gcov[1][0] = gcov[0][1]

    gcov[1][3] = -a*(1+2*r/Sigma)*sth2
    gcov[3][1] = gcov[1][3]
    
    return gcov

# contravariant metric in KS
def gconKS(spin,r,th):

    a = spin
    a2 = a**2
    r2 = r**2
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
        
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2
    Pi = (r2+a2)**2 - a2*Delta*sth2
    
    if isinstance(r,np.ndarray):
        gcon = np.zeros(np.hstack(((4,4),r.shape)))
    else:
        gcon = np.zeros((4,4))
    
    gcon[0][0] = -(1+2*r/Sigma)
    gcon[1][1] = Delta/Sigma
    gcon[2][2] = 1./Sigma
    gcon[3][3] = 1./(Sigma*sth2)
    
    gcon[0][3] = 0
    gcon[3][0] = gcon[0][3]
    
    gcon[0][1] = 2*r/Sigma
    gcon[1][0] = gcon[0][1]
    
    gcon[1][3] = a/Sigma   
    gcon[3][1] = gcon[1][3]
    
    return gcon

# KS metric determinant
def gdetKS(a,r,th):
    return np.sin(th)*(r**2 + a**2*np.cos(th)**2)

# lower indices in KS
def lowerKS(a0, a1, a2, a3, spin, r, th):
    g = gcovKS(spin,r,th)    
    a0_l = g[0][0]*a0 + g[0][1]*a1 + g[0][2]*a2 + g[0][3]*a3
    a1_l = g[1][0]*a0 + g[1][1]*a1 + g[1][2]*a2 + g[1][3]*a3
    a2_l = g[2][0]*a0 + g[2][1]*a1 + g[2][2]*a2 + g[2][3]*a3
    a3_l = g[3][0]*a0 + g[3][1]*a1 + g[3][2]*a2 + g[3][3]*a3
    
    return (a0_l, a1_l, a2_l, a3_l)

# raise indices in KS
def raiseKS(a0, a1, a2, a3, spin, r, th):
    g = gconKS(spin,r,th)    
    a0_u = g[0][0]*a0 + g[0][1]*a1 + g[0][2]*a2 + g[0][3]*a3
    a1_u = g[1][0]*a0 + g[1][1]*a1 + g[1][2]*a2 + g[1][3]*a3
    a2_u = g[2][0]*a0 + g[2][1]*a1 + g[2][2]*a2 + g[2][3]*a3
    a3_u = g[3][0]*a0 + g[3][1]*a1 + g[3][2]*a2 + g[3][3]*a3
    
    return (a0_u, a1_u, a2_u, a3_u)
          
# compute 4 velocity from KS relative velocity primitives
def conv_vel_KS(u1_velr, u2_velr, u3_velr, spin, r, th):

    gcon = gconKS(spin,r,th)
    gcov = gcovKS(spin,r,th)
    
    utilde = (0,u1_velr, u2_velr, u3_velr)
    qsq = 0
    for i in (1,2,3):
        for j in (1,2,3):
            qsq += utilde[i]*utilde[j]*gcov[i][j]
            
    gamma2 = (1 + qsq)
    alpha2 = -1 / gcon[0][0]
    alphagamma = np.sqrt(alpha2*gamma2)
    
    u0 = 0       - alphagamma * gcon[0][0]
    u1 = u1_velr - alphagamma * gcon[0][1]
    u2 = u2_velr - alphagamma * gcon[0][2]
    u3 = u3_velr - alphagamma * gcon[0][3]
    
    return (u0,u1,u2,u3)

# compute KS relative velocity primitives from 4 velocity
def invconv_vel_KS(u0, u1, u2, u3, spin, r, th):

    gcon = gconKS(spin,r,th)
    
    v1 = u1 - u0 * gcon[0][1]/gcon[0][0]
    v2 = u2 - u0 * gcon[0][2]/gcon[0][0]
    v3 = u3 - u0 * gcon[0][3]/gcon[0][0]
    
    return (v1,v2,v3)

# re-compute u0 from u1,u2,u3
def fill_ut_in_ucon_KS(u1,u2,u3,spin,r,th):

    #-1 = g00 u0^2 + 2 gi0 ui u0 + gij ui uj
     
    gcov = gcovKS(spin,r,th)
    
    uu = (0,u1, u2, u3)
    
    a = gcov[0][0]
    b = 0
    c = 1
    
    for i in (1,2,3):
        b += uu[i]*gcov[i][0]
        for j in (1,2,3):
            c += uu[i]*uu[j]*gcov[i][j]
    
    delta = b*b - a*c
    u0 = (-b-np.sqrt(delta))/a        
    
    return u0
    
# compute KS Lorentz factor
def lorentz_KS(u0, spin, r, th):
    gcon = gconKS(spin,r,th)
    lorentz = u0 / np.sqrt(-gcon[0][0])
    return lorentz

##################################
# Boyer-Lindquist
##################################

# covariant metric in BL
def gcovBL(spin,r,th):

    a = spin
    a2 = a**2
    r2 = r**2
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
        
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2
    Pi = (r2+a2)**2 - a2*Delta*sth2
    
    if isinstance(r,np.ndarray):
        gcov = np.zeros(np.hstack(((4,4),r.shape)))
    else:
        gcov = np.zeros((4,4))
        
    gcov[0][0] = -(1-2*r/Sigma)
    gcov[1][1] = Sigma/Delta
    gcov[2][2] = Sigma
    gcov[3][3] = Pi*sth2/Sigma
    
    gcov[0][3] = -2*r*a*sth2 / Sigma
    gcov[3][0] = gcov[0][3]
            
    return gcov

# contravariant metric in BL
def gconBL(spin,r,th):

    a = spin
    a2 = a**2
    r2 = r**2
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
        
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2
    Pi = (r2+a2)**2 - a2*Delta*sth2
    
    if isinstance(r,np.ndarray):
        gcon = np.zeros(np.hstack(((4,4),r.shape)))
    else:
        gcon = np.zeros((4,4))
    
    gcon[0][0] = -Pi/(Delta*Sigma)
    gcon[1][1] = Delta/Sigma
    gcon[2][2] = 1./Sigma
    gcon[3][3] = (Delta - a2*sth2)/(Delta*Sigma*sth2)
    
    gcon[0][3] = -2*a*r/(Delta*Sigma)
    gcon[3][0] = gcon[0][3]
    
    return gcon

# BL metric determinant #??
def gdetBL(a,r,th):
    #return 0.5*np.sin(th)*(a**2 + 2*r**2 + a**2*np.cos(2*th)) #??  
    return np.sin(th)*(r**2 + a**2*np.cos(th)**2)
    
# lower indices in BL
def lowerBL(a0, a1, a2, a3, spin, r, th):
    g = gcovBL(spin,r,th)    
    a0_l = g[0][0]*a0 + g[0][1]*a1 + g[0][2]*a2 + g[0][3]*a3
    a1_l = g[1][0]*a0 + g[1][1]*a1 + g[1][2]*a2 + g[1][3]*a3
    a2_l = g[2][0]*a0 + g[2][1]*a1 + g[2][2]*a2 + g[2][3]*a3
    a3_l = g[3][0]*a0 + g[3][1]*a1 + g[3][2]*a2 + g[3][3]*a3
    
    return (a0_l, a1_l, a2_l, a3_l)

# raise indices in BL
def raiseBL(a0, a1, a2, a3, spin, r, th):
    g = gconBL(spin,r,th)    
    a0_u = g[0][0]*a0 + g[0][1]*a1 + g[0][2]*a2 + g[0][3]*a3
    a1_u = g[1][0]*a0 + g[1][1]*a1 + g[1][2]*a2 + g[1][3]*a3
    a2_u = g[2][0]*a0 + g[2][1]*a1 + g[2][2]*a2 + g[2][3]*a3
    a3_u = g[3][0]*a0 + g[3][1]*a1 + g[3][2]*a2 + g[3][3]*a3
    
    return (a0_u, a1_u, a2_u, a3_u)

# lower indices in BL for a 22 tensor
def lower22(Fcon,gcov):
    #gcov = gconBL(spin,r,th)    
    Fcov = np.zeros(Fcon.shape)
    
    # TODO faster? 
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    Fcov[i][j] += gcov[i][k]*gcov[j][l]*Fcon[k][l]
    return Fcov
                        
# compute 4 velocity from BL relative velocity primitives
def conv_vel_BL(u1_velr, u2_velr, u3_velr, spin, r, th):

    gcon = gconBL(spin,r,th)
    gcov = gcovBL(spin,r,th)
    
    utilde = (0,u1_velr, u2_velr, u3_velr)
    qsq = 0
    for i in (1,2,3):
        for j in (1,2,3):
            qsq += utilde[i]*utilde[j]*gcov[i][j]
            
    gamma2 = (1 + qsq)
    alpha2 = -1 / gcon[0][0]
    alphagamma = np.sqrt(alpha2*gamma2)
    
    u0 = 0       - alphagamma * gcon[0][0]
    u1 = u1_velr - alphagamma * gcon[0][1]
    u2 = u2_velr - alphagamma * gcon[0][2]
    u3 = u3_velr - alphagamma * gcon[0][3]
    
    return (u0,u1,u2,u3)

# compute KS relative velocity primitives from 4 velocity
def invconv_vel_BL(u0, u1, u2, u3, spin, r, th):

    gcon = gconBL(spin,r,th)
    
    v1 = u1 - u0 * gcon[0][1]/gcon[0][0]
    v2 = u2 - u0 * gcon[0][2]/gcon[0][0]
    v3 = u3 - u0 * gcon[0][3]/gcon[0][0]
    
    return (v1,v2,v3)

# re-compute u0 from u1,u2,u3
def fill_ut_in_ucon_BL(u1,u2,u3,spin,r,th):

    #-1 = g00 u0^2 + 2 gi0 ui u0 + gij ui uj
     
    gcov = gcovBL(spin,r,th)
    
    uu = (0,u1, u2, u3)
    
    a = gcov[0][0]
    b = 0
    c = 1
    
    for i in (1,2,3):
        b += uu[i]*gcov[i][0]
        for j in (1,2,3):
            c += uu[i]*uu[j]*gcov[i][j]
    
    delta = b*b - a*c
    u0 = (-b-np.sqrt(delta))/a        
    
    return u0
       
# compute BL Lorentz factor
def lorentz_BL(u0, spin, r, th):
    gcon = gconBL(spin,r,th)
    lorentz = u0 / np.sqrt(-gcon[0][0])
    return lorentz

##################################
# Transformations
##################################
    
# transformation from KS 2 BL
def dxdx_ks2bl(spin, r, th):
    a = spin
    a2 = a**2
    Delta = r**2 - 2*r + a**2
    Sigma = r**2 + a**2 * np.cos(th)**2
    sth2 = np.sin(th)**2
    
    if isinstance(r,np.ndarray):
        dxdx = np.zeros(np.hstack(((4,4),r.shape)))
    else:
        dxdx = np.zeros((4,4))
    
    dxdx[0][0] = 1.
    dxdx[1][1] = 1.
    dxdx[2][2] = 1. 
    dxdx[3][3] = 1.
    
    dxdx[0][1] = 2*r/Delta
    dxdx[3][1] = a/Delta
    
    return dxdx
    
# transformation from BL 2 KS
def dxdx_ks2bl(spin, r, th):
    a = spin
    a2 = a**2
    Delta = r**2 - 2*r + a**2
    Sigma = r**2 + a**2 * np.cos(th)**2
    sth2 = np.sin(th)**2
    
    if isinstance(r,np.ndarray):
        dxdx = np.zeros(np.hstack(((4,4),r.shape)))
    else:
        dxdx = np.zeros((4,4))
    
    dxdx[0][0] = 1.
    dxdx[1][1] = 1.
    dxdx[2][2] = 1. 
    dxdx[3][3] = 1.
    
    dxdx[0][1] = -2*r/Delta
    dxdx[3][1] = -a/Delta
    
    return dxdx
   
def trans_cov_ks2bl(a0_ks,a1_ks,a2_ks,a3_ks,spin,r,th):
    """u2^i (BL) = A^i_j u1^j (KS) """
    
    A = dxdx_ks2bl(spin,r,th) 
       
    a0_bl = A[0][0]*a0_ks + A[0][1]*a1_ks + A[0][2]*a2_ks + A[0][3]*a3_ks
    a1_bl = A[1][0]*a0_ks + A[1][1]*a1_ks + A[1][2]*a2_ks + A[1][3]*a3_ks
    a2_bl = A[2][0]*a0_ks + A[2][1]*a1_ks + A[2][2]*a2_ks + A[2][3]*a3_ks
    a3_bl = A[3][0]*a0_ks + A[3][1]*a1_ks + A[3][2]*a2_ks + A[3][3]*a3_ks
    
    return (a0_bl, a1_bl, a2_bl, a3_bl)    
    
def trans_cov_bl2ks(a0_bl,a1_bl,a2_bl,a3_bl,spin,r,th):
    """u2^i (ks) = A^i_j u1^j (bl) """
    
    A = dxdx_bl2ks(spin,r,th)
        
    a0_ks = A[0][0]*a0_bl + A[0][1]*a1_bl + A[0][2]*a2_bl + A[0][3]*a3_bl
    a1_ks = A[1][0]*a0_bl + A[1][1]*a1_bl + A[1][2]*a2_bl + A[1][3]*a3_bl
    a2_ks = A[2][0]*a0_bl + A[2][1]*a1_bl + A[2][2]*a2_bl + A[2][3]*a3_bl
    a3_ks = A[3][0]*a0_bl + A[3][1]*a1_bl + A[3][2]*a2_bl + A[3][3]*a3_bl
    
    return (a0_ks, a1_ks, a2_ks, a3_ks)          
    
