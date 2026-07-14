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

Definitions
-----------

Flux Coordinates: [rho, theta, zeta]
    where rho = sqrt(s) = sqrt(psi/psi_edge)


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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eq = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eq = None

    def default_config(self):
        config = super().default_config()
        config['wout_file']           = None
        config['emissivity_scale']    = 1.0
        config['temperature_scale']   = 1.0
        config['temperature_e_scale'] = 1.0
        config['velocity_scale']      = 1.0
        return config
        
    # we use self.eq because we want to use the equilibrium in other methods
    def initialize_vmec(self, wout=None):
        if wout is None:
            wout = self.param['wout_file']
        self.eq = VMECIO.load(wout)
    
    def flx_from_car(self, point_car):
        if self.eq is None:
            self.initialize_vmec()

        point_flx = self.eq.map_coordinates(
            point_car, 
            inbasis = ("X", "Y", "Z"), 
            outbasis = ("rho", "theta", "zeta"),
        )
        return point_flx

    def car_from_flx(self, point_flx):
        if self.eq is None:
            self.initialize_vmec()

        point_car = self.eq.map_coordinates(
            point_flx,
            inbasis = ("rho", "theta", "zeta"),
            outbasis = ("X", "Y", "Z"),
        )
        return point_car

    def flx_from_cyl(self, point_cyl):
        if self.eq is None:
            self.initialize_vmec()

        point_flx = self.eq.map_coordinates(
            point_cyl, 
            inbasis = ("R", "phi", "Z"),
            outbasis = ("rho", "theta", "zeta"),
        )
        return point_flx

    def cyl_from_flx(self, point_flx):
        if self.eq is None:
            self.initialize_vmec()

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
        return point_car

    # DESC uses rho directly; stelltools returns s so rho=sqrt(s)
    def rho_from_car(self, point_car):
        point_flx = self.flx_from_car(point_car)
        return point_flx[:, 0]

    def bundle_generate(self, bundle_input):
        self.log.debug('Starting bundle_generate')

        self.initialize_vmec()
        
        profiler.start("Bundle Input Generation")
        m = bundle_input['mask']

        # Attempt to generate the specified number of bundles, but throw out
        # bundles that are outside the last closed flux surface.

        profiler.start("Fluxspace from Realspace")
        points = bundle_input['origin'][m]
        point_flx_temp = self.flx_from_car(points)
        # Convert the jax ndarray into an editable numpy mutable ndarray
        rho = np.asarray(point_flx_temp[:, 0]).copy()
        profiler.stop("Fluxspace from Realspace")


        profiler.start("Realspace from Fluxspace")
        # DESC's coordinate transformation is unreliable for points far from LCFS
        # using the round-trip error to filter out points outside LCFS
        point_car_check = self.car_from_flx(point_flx_temp)
        error = np.linalg.norm(point_car_check - points, axis=1)
        rho[(~np.isfinite(rho)) | (rho >= 1.0) | (error > 1e-2)] = np.nan
        profiler.stop("Realspace from Fluxspace")

        
        # evaluate emissivity, temperature and velocity at each bundle location.
        bundle_input['temperature'][m]   = self.get_temperature(rho)   * self.param['temperature_scale']
        bundle_input['temperature_e'][m] = self.get_temperature_e(rho) * self.param['temperature_e_scale']
        bundle_input['emissivity'][m]    = self.get_emissivity(rho)    * self.param['emissivity_scale']
        bundle_input['velocity'][m]      = self.get_velocity(rho)      * self.param['velocity_scale']
        
        fintest = np.isfinite(bundle_input['temperature'])
        m &= fintest
        
        profiler.stop("Bundle Input Generation")

        return bundle_input
