# -*- coding: utf-8 -*-
"""
Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
  - Edited by Leila Alston <leilaalston07@gmail.com>

Description
-----------
A plasma source based on a VMEC equilibrium.
Rewritten to perform all coordinate transformations using DESC.
"""

import numpy as np

from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.sources._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric

import desc
from desc.vmec import VMECIO
from desc.grid import Grid

@dochelper
class XicsrtPlasmaVmec(XicsrtPlasmaGeneric):

    def default_config(self):
        config = super().default_config()
        config['wout_file']         = None
        config['emissivity_scale']  = 1.0
        config['temperature_scale'] = 1.0
        config['velocity_scale']    = 1.0
        return config
        
    # we use self.eq because we want to use the equilibrium in other methods
    def initialize_vmec(self, wout=None):
        if wout is None:
            wout = self.param['wout_file']
        self.eq = VMECIO.load(wout)
    
    def flx_from_car(self, point_car):
        point_flx = self.eq.map_coordinates(
            point_car, 
            inbasis = ("X", "Y", "Z"), 
            outbasis = ("rho", "theta", "zeta"),
        )
        return point_flx

    def car_from_flx(self, point_flx):
        point_car = self.eq.map_coordinates(
            point_flx,
            inbasis = ("rho", "theta", "zeta"),
            outbasis = ("X", "Y", "Z"),
        )
        return point_car

    def flx_from_cyl(self, point_cyl):
        point_flx = self.eq.map_coordinates(
            point_cyl, 
            inbasis = ("R", "phi", "Z"),
            outbasis = ("rho", "theta", "zeta"),
        )
        return point_flx

    def cyl_from_flx(self, point_flx):
        point_cyl = self.eq.map_coordinates(
            point_flx, 
            inbasis = ("rho", "theta", "zeta"),
            outbasis = ("R", "phi", "Z"),
        )
        return point_cyl

    def cyl_from_car(self, point_car):
        point_cyl = self.eq.map_coordinates(
            point_car, 
            inbasis = ("X", "Y", "Z"),
            outbasis = ("R", "phi", "Z"),
        )
        return point_cyl

    def car_from_cyl(self, point_cyl):
        point_car = self.eq.map_coordinates(
            point_cyl, 
            inbasis = ("R", "phi", "Z"),
            outbasis = ("X", "Y", "Z"),
        )

    # DESC uses rho directly; stelltools returns s so rho=sqrt(s)
    def rho_from_car(self, point_car):
        point_flx = self.flx_from_car(point_car)
        return point_flx[0]

    def bundle_generate(self, bundle_input):
        self.log.debug('Starting bundle_generate')

        self.initialize_vmec()
        
        profiler.start("Bundle Input Generation")
        m = bundle_input['mask']

        # Attempt to generate the specified number of bundles, but throw out
        # bundles that our outside of the last closed flux surface.
        #
        # Currently DESC coordinate inversion may only handle one point at a time,
        # so a loop is required. This will be improved eventually.
        rho = np.zeros(len(m[m]))
        for ii in range(len(m[m])):
            # convert from cartesian coordinates to normalized radial coordinate.
            profiler.start("Fluxspace from Realspace")
            try:
                rho[ii] = self.rho_from_car(bundle_input['origin'][m][ii,:])
            except Exceptions:
                rho[ii] = np.nan
            profiler.stop("Fluxspace from Realspace")
        
        # evaluate emissivity, temperature and velocity at each bundle location.
        bundle_input['temperature'][m] = self.get_temperature(rho) * self.param['temperature_scale']
        bundle_input['emissivity'][m]  = self.get_emissivity(rho)  * self.param['emissivity_scale']
        bundle_input['velocity'][m]    = self.get_velocity(rho)    * self.param['velocity_scale']
        
        fintest = np.isfinite(bundle_input['temperature'])
        m &= fintest
        
        profiler.stop("Bundle Input Generation")

        return bundle_input
