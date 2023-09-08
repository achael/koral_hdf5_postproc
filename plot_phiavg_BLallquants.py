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
from read_phiavg_BLallquants import *
import h5py


##general preamble
matplotlib.rcdefaults()
matplotlib.rc('font',**{'family':'serif','size':16})
os.environ['PATH'] = os.environ['PATH'] + ':/opt/local/bin'
os.environ['PATH']
plt.close('all')
plt.rc('text', usetex=True)

#NCOLS = 2  #number of models
#NROWS = 1  #number of quantities to plot
LABELSIZE=14
LABELSIZE2=10
NAPHICONTOUR = 20

METRIC_OUT = 'BL'
PLOT_APHI = True
PLOT_SIGMA = True
WSPACE = -.5
PLOTLABEL = True

# region to plot
xlim=50
zlim1=-10
zlim2=50 

zlimplot1 = zlim1
zlimplot2 = zlim2
xlimplot = xlim

xticks = np.arange(0,xlim+10,10)
yticks = np.arange(zlim1, zlim2+10, 10)
xticks_min = np.arange(5,xlim,10)
yticks_min = np.arange(zlim1+5,zlim2,10)

#xticks = [0,5,10,15,20]
#yticks = [-10,-5,0,5,10]
#xticks_min = [2.5,7.5,12.5,17.5]
#yticks_min = [-7.5,-2.5,2.5,7.5]

#files & paths
pathout='./'

filep94='./disk24_ipole_tavg11000-16000.h5'
filep9='./a.9_ipole_tavg50000-100000.h5'


files = np.array([[filep94,filep9]])
spins = np.array([[0.9375,0.9]])
labels = np.array([[r'Chael+ 2019 ($a=0.94$)',r'Narayan+ 2021 ($a=0.9$)']])
tag = None

NROWS = files.shape[0]
NCOLS = files.shape[1]

plotfields = ['logsigma2']

colmaps = {field:'Spectral' for field in plotfields}
colmaps['bflux'] = 'magma'
colmaps['bpol'] = 'magma'

cbarlabels = {
'omegafield_r':r'$\Omega_{F}$ $[t^{-1}_{\rm g}]$',
'omegafield_th':r'$\Omega_{F}$ $[t^{-1}_{\rm g}]$',
'omegafluid':r'$v^{\phi}$',
'femag':r'$\sqrt{-g} F_E$',
'femag_norm':r'$\sqrt{-g} \mathcal{J}^r_\mathcal{E}$ [norm]',

'femag_prims':r'$-\sqrt{-g} T^r_{t \;\; \mathrm{EM}}$ (prims)',
'femag_prims_norm':r'$-\sqrt{-g} T^r_{t \;\; \mathrm{EM}}$ (prims,norm)',

'femag_starf':r'$-\sqrt{-g} T^r_{t \;\; \mathrm{EM}}$ ($\star F$)',
'femag_starf_norm':r'$-\sqrt{-g} T^r_{t \;\; \mathrm{EM}}$ ($\star F$,norm)',

'fehd':r'$-\sqrt{-g} T^r_{t \;\; \mathrm{HD}}$',
'fehd_norm':r'$-\sqrt{-g} T^r_{t \;\; \mathrm{HD}}$ (norm)',

'bratio':r'$\frac{B^\phi}{B^r}$',
'bratio_sign':r'$\mathrm{sign}\left[\frac{B^\phi}{B^r}\right]$',
'bflux':r'$-\sqrt{-g} |B^r|$',
'bpol':r'$\log B_{\rm pol}$',
'logsigma2':r'$\log\sigma$',
'logbeta2':r'$\log\beta$'
}
plotranges = {
'omegafield_r':[-0.075,0.075],
'omegafield_th':[-0.05,0.05],
'omegafluid':[-2,2],

'fehd':[-4,4],
'fehd_norm':[-1,1],

'femag':[-.1,.1],
'femag_norm':[-1,1],
'femag_prims':[-1,1],
'femag_prims_norm':[-.05,.05],
'femag_starf':[-1,1],
'femag_starf_norm':[-1,1],
'bratio':[-0.1,0.1],
'bratio_sign':[-1,1],
'bflux':[0,10],
'bpol':[-3,2],
'logsigma2':[-3,3],
'logbeta2':[-3,3]
}


def main():
    ##make one plot per file
    plt.close('all')
    for jj in range(NROWS):
        for ii in range(NCOLS):
            print ('-------',ii,jj,'--------')
            filein = files[jj,ii]
            label = labels[jj,ii]
            spin = spins[jj,ii]
            
            horiz = 1 + np.sqrt(1-spin**2)    
            omegaH = spin / (2*horiz)
                
            # get data
            datdict = unpack_hdf5_blallquants(filein)
            
            # plot data   
            for kk, field in enumerate(plotfields):
            
                fig = plt.figure(kk,figsize=(4*NCOLS,3*NROWS), dpi=600)
        
                colmap = colmaps[field]
                cbarlabel = cbarlabels[field]
                plotrange = plotranges[field]

                xsign = 1
                col = 1
                slope = (5./50.)
                ax0 = plt.subplot2grid((NROWS, NCOLS), (jj,ii),aspect='equal')
                ax = ax0
                                   
#                if(ii==0):
#                    xsign = 1
#                    col = 1
#                    slope = (5./50.)
#                    ax0 = plt.subplot2grid((NROWS, NCOLS), (jj,col),aspect='equal')
#                    ax = ax0
#                elif(ii==1): 
#                    xsign=-1
#                    col = 0
#                    slope = (5./50.)
#                    ax1 = plt.subplot2grid((NROWS, NCOLS), (jj,col),aspect='equal')    
#                    ax = ax1
                            
                ###################################################
                (grid_x,grid_z,grid_data) = my_griddata(datdict, field, reggrid=False)

                # normalize
                if 'norm' in field:
                    grid_r = np.sqrt(grid_x**2 + grid_z**2)
                    #normfac = np.max(np.abs(grid_data))
                    
                    norm_r_min = 1.1*horiz
                    norm_r_max = xlim
                    normmask = (grid_r>norm_r_min) * (grid_r < norm_r_max)
                    
                    
                    normfac = np.max(np.abs(grid_data[normmask]))
                    #normfac_r = grid_r[grid_r>norm_r][np.argmax(np.abs(grid_data[grid_r>norm_r]))]
                    #print(normfac_r, normfac)
                    grid_data = grid_data/normfac
                            
                
                im = ax.pcolormesh(xsign*grid_x, grid_z, grid_data, 
                                   vmin=plotrange[0],vmax=plotrange[1],
                                   shading='gouraud',
                                   cmap=colmap,rasterized=True)                               

                # bfield contours
                if PLOT_APHI:
                    grid_x,grid_z,grid_phi = my_griddata(datdict,'Aphi_th',verbose=False)
                    grid_phipos = np.ma.masked_where(grid_phi > 0, grid_phi)
                    grid_phineg = np.ma.masked_where(grid_phi < 0, grid_phi)
               
                    levels = np.linspace(np.min(grid_phipos), np.max(grid_phipos), NAPHICONTOUR)
                    cntr20=ax.contour(xsign*grid_x, grid_z, grid_phipos, linewidths=.5, colors= "w", levels=levels, linestyles='solid')

                    levels = np.linspace(np.min(grid_phineg), np.max(grid_phineg), NAPHICONTOUR)
                    cntr20=ax.contour(xsign*grid_x, grid_z, grid_phineg, linewidths=.5, colors= "w", levels=levels, linestyles='dashed')
                
                # sigma=1 contours
                if PLOT_SIGMA:
                    grid_x,grid_z,grid_sigma = my_griddata(datdict,'sigma2',verbose=False)
                    clevels = [1]
                    cntr21=ax.contour(xsign*grid_x, grid_z, grid_sigma, linewidths=2, levels=clevels, colors= "c")
                
                # black hole and axis patches
                circle1=plt.Circle((0,0),1.00001*horiz, color='k',zorder=10)   
                ax.add_artist(circle1)
                xxc = 0
                
                tri1=plt.Polygon([(xxc,0),(xsign*xxc,zlim2),(xsign*(xxc+zlim2*slope),zlim2)], closed=True,color='k',zorder=10)
                ax.add_artist(tri1)
                tri2=plt.Polygon([(xxc,0),(xsign*xxc,zlim1),(xsign*(xxc+abs(zlim1)*slope),zlim1)], closed=True,color='k',zorder=10)
                ax.add_artist(tri2)

                # title and ticks
                if field=='bratio_sign' or PLOTLABEL: ax.set_title(label,fontsize=LABELSIZE2)
                else: ax.set_title(label,fontsize=LABELSIZE, color='w')
                
                ax.tick_params(labelsize=LABELSIZE)

                ax.set_xticks(xticks)
                ax.set_xticks(xticks_min, labels=[], minor=True)

                ax.set_yticks(yticks_min,labels=[], minor=True)
                
                if ii==0:
                    ax.set_yticks(yticks,labels=yticks)
                    ax.set_ylabel(r'$z/r_{\rm g}$',fontsize=LABELSIZE)
                else:
                    ax.set_yticks(yticks,labels=[])
                #if ii==1:    
                #    if field=='femag_norm': ax.set_xlabel(r'$x/r_{\rm g}$',fontsize=LABELSIZE)
                #    else: ax.set_xlabel(r'$x/r_{\rm g}$',fontsize=LABELSIZE, color='w')
                ax.set_ylim(zlimplot1,zlimplot2)
                ax.set_xlim(0,xlimplot)
                ax.set_xlabel(r'$x/r_{\rm g}$',fontsize=LABELSIZE)                                        

                # color bar
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes("right", size="5%", pad=0.05)
#                if ii==1:
                if ii<NCOLS-1:
                    cax.axis('off')
                
                else:
                    cbar = fig.colorbar(im, cax=cax, orientation="vertical",format='%2.4g')#,ticks=[-1,0,1])    
                    cax.set_title(cbarlabel,fontsize=int(8),pad=10)    
                    cbar.ax.yaxis.set_label_position('right')
                    cbar.ax.tick_params(labelsize=LABELSIZE) 

    # save
    for kk, field in enumerate(plotfields):
        fig = plt.figure(kk)
        outname = 'phiavg_' + plotfields[kk]
        if tag is not None:
            outname += tag
        fileout = pathout+outname + '.pdf'
                
        plt.tight_layout()
        fig.subplots_adjust(wspace=WSPACE,hspace=0.)
        plt.savefig(fileout, dpi=100,bbox_inches="tight")

if __name__=='__main__':
    main()
