# Andrew Chael, September 2026
# shell-integrated fluxes as a function of r, from phi-averaged koral hdf5 files
# assumes axisymmetric metric
#
# Every flux is an integral over a sphere of constant r,
#
#   F(r) = int_0^2pi int_0^pi f sqrt(-g) dth dph = 2pi sum_j <f>_ph sqrt(-g) dth_j
#
# where <f>_ph is a quantity stored in a phi-averaged file.  The phi integral is
# exact for an axisymmetric metric; dth_j comes from finite differences of the
# cell-centre theta, which reproduces KORAL's own volume element (dth/dx2)*dx2 on
# any grid uniform in x2, including the cylindrified polar cells of JETCOORDS.
#
# To add a flux, write a function of (dat, shell) returning one value per radius
# and register it with @register_flux; see mdot() below.

import numpy as np
from collections import OrderedDict, namedtuple
if __package__: # imported as part of the package, from another directory
    from .metricKS import *
else:           # run as a script, or imported with this directory on sys.path
    from metricKS import *


class Shell(object):
    """The (r, theta) grid of a phi-averaged file, set up for shell integrals"""

    def __init__(self, dat):
        """dat -- a simdata2D, from read_koral_hdf52D"""
        if dat.metric not in ['KS','BL']:
            raise Exception("metric must be 'KS' or 'BL', got '%s' in %s"
                            % (dat.metric, dat.filename))
        r, th = dat.r, dat.th
        if not np.allclose(r, r[:,:1], rtol=1.e-12, atol=0):
            raise Exception("r varies along theta in %s, cannot integrate on shells"
                            % dat.filename)

        self.spin = float(dat.spin)
        self.metric = dat.metric
        self.horiz = 1 + np.sqrt(1 - self.spin**2)
        self.r = r[:,0]
        self.th = th

        # sqrt(-g) is the same function in KS and BL
        self.gdet = gdetKS(self.spin, r, th)
        # covariant metric, for lowering indices of the stored vectors/tensors
        if self.metric == 'KS':
            self.gcov = gcovKS(self.spin, r, th)
        else:
            self.gcov = gcovBL(self.spin, r, th)
        # (dth/dx2)*dx2 at the cell centres, assuming uniform x2
        self.dth = np.gradient(th, axis=1)

        # 2pi for the phi integral of a phi-averaged quantity
        self.weight = 2*np.pi*self.gdet*self.dth

    def integrate(self, integrand, mask=None):
        """Integrate a phi-averaged quantity over each sphere, optionally only
        over the cells where mask is True.  Returns one value per radius."""
        w = self.weight if mask is None else np.where(mask, self.weight, 0.)
        return np.sum(integrand*w, axis=1)


###############################################################################
# registry of fluxes
###############################################################################

FluxDef = namedtuple('FluxDef', ['name', 'description', 'needs', 'func', 'notes'])
FLUXES = OrderedDict()

def register_flux(name, description, needs, notes=()):
    """Decorator adding a flux to FLUXES.

    name        -- column name in the output table
    description -- one line for the output header
    needs       -- stored quantities the flux reads: names from dat.field_names(),
                   read with dat.field(name).  Checked before computing.
    notes       -- optional caveats, one header line each, below the description
    """
    def deco(func):
        FLUXES[name] = FluxDef(name, description, list(needs), func, list(notes))
        return func
    return deco


###############################################################################
# helpers
###############################################################################

# stored T^{r nu}: only the upper triangle T^{mu nu}, mu <= nu, is written
TRNU = ['T01', 'T11', 'T12', 'T13']
MHD_PARTS = ('hd', 'mag')

def needs_T(parts=MHD_PARTS):
    """Stored quantities T_r_t reads for the given stress-tensor parts"""
    return ['%s_%s' % (t, p) for p in parts for t in TRNU]


def T_r_t(dat, shell, parts=MHD_PARTS):
    """Mixed T^r_t = g_{t nu} T^{r nu}, summed over the stress-tensor parts.

    Lowering with the phi- and time-independent metric commutes with the
    averages, so this is exactly the average of T^r_t.
    """
    g = shell.gcov
    Trt = 0.
    for nu in range(4):
        Tr = sum(dat.field('%s_%s' % (TRNU[nu], p)) for p in parts)
        Trt = Trt + g[0][nu]*Tr
    return Trt


def jet_mask(dat, shell, parts=MHD_PARTS):
    """Jet region of Paper V: (beta gamma)^2 = (-T^r_t/(rho u^r))^2 - 1 >= 1.

    Built from whatever averages are in dat, so for a time-average this is the
    cut on the time- and phi-averaged T^r_t and rho u^r that the spec asks for;
    for a single phi-averaged dump it is that dump's own cut.  Cells with
    rho u^r = 0 and T^r_t != 0 count as jet.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        bgsq = (-T_r_t(dat, shell, parts)/dat.field('rhou1'))**2 - 1
    return bgsq >= 1


###############################################################################
# fluxes
###############################################################################

@register_flux('Mdot', 'mass accretion rate, -int rho u^r sqrt(-g), positive inward',
               needs=['rhou1'])
def mdot(dat, shell):
    return -shell.integrate(dat.field('rhou1'))


@register_flux('Phi', 'magnetic flux, 0.5 int |B^r| sqrt(-g), Heaviside-Lorentz',
               needs=['absB1'])
def phi(dat, shell):
    return 0.5*shell.integrate(dat.field('absB1'))


@register_flux('Pout', 'total outflow power, -int (T^r_t + rho u^r) sqrt(-g), MHD only',
               needs=['rhou1'] + needs_T())
def pout(dat, shell):
    return -shell.integrate(T_r_t(dat, shell) + dat.field('rhou1'))


@register_flux('Pjet', 'jet power, Pout over the (beta gamma)^2 >= 1 cut on the averages, MHD only',
               needs=['rhou1'] + needs_T(),
               notes=['Pjet is the power of BOTH jets (north + south); one jet carries ~Pjet/2',
                      '(beta gamma)^2 squares -T^r_t/(rho u^r), so near the BH (r <~ 10) the cut also '
                      'takes in magnetized inflow where rho u^r < 0 but -T^r_t > 0'])
def pjet(dat, shell):
    mask = jet_mask(dat, shell)
    return -shell.integrate(T_r_t(dat, shell) + dat.field('rhou1'), mask=mask)


def compute_fluxes(dat, names=None):
    """Compute the requested fluxes (default: all registered) on every shell.

    Returns an OrderedDict name -> array over r, starting with 'r' itself.
    """
    if names is None:
        names = list(FLUXES.keys())
    unknown = [n for n in names if n not in FLUXES]
    if unknown:
        raise Exception("unknown flux(es) %s -- available: %s"
                        % (unknown, ', '.join(FLUXES.keys())))

    stored = set(dat.field_names())
    missing = sorted(set(q for n in names for q in FLUXES[n].needs) - stored)
    if missing:
        raise Exception("%s is missing quantities %s needed for the fluxes"
                        % (dat.filename, missing))

    shell = Shell(dat)
    out = OrderedDict([('r', shell.r)])
    for n in names:
        out[n] = FLUXES[n].func(dat, shell)
    return out
