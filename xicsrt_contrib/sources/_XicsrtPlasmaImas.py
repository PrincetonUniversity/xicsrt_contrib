# -*- coding: utf-8 -*-
"""
.. Authors
    Nathan Bartlett <nbb0011@auburn.edu>
    Novimir Pablant <npablant@pppl.gov>

Description
-----------
A plasma source based on ITER IMAS data.
"""

import numpy as np
import logging

from xicsrt import xicsrt_io
from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.sources._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric
import os
import pathlib
from scipy import interpolate

@dochelper
class XicsrtPlasmaImas(XicsrtPlasmaGeneric):
    """
    Description
    -----------
    A plasma source based on ITER IMAS data.

    Programming Notes
    -----------------

    .. ToDo:
        There are several issues with the `XicsrtPlasmaImas` object that need to be
        resolved. See the class header for more information

    - There is an inconsistency between 1d_psi_data (or psi_axis) and the
      2d_psi_data. This means that we don't know what the maximum value of the
      2d_psi_data should be, and therefore we cannot normalize to s or rho
      correctly. For now the maximum grid point is used. Once this inconsistency
      is resolved, the code should use psi_axis and psi_boundary instead.
    - This code uses a number of hard-coded regions to define the ITER plasma
      core and exclude the private flux region from calculations. This is fine
      for iter but A more general solution is preferred incase we ever want
      to use this object with a different tokamak.
    - The forward and reverse algorithms for conversion between rho and R,Z
      are not consistent. A full round conversion results in errors on the
      order of 1e-5. The problem is probably in car_from_flux, which is
      currently only used for plotting. This problem needs to examined and
      fixed; a full round calculation should be accurate to near machine
      precision.

    Overall this object needs some major restructuring for readability.
    Most importantly the imas data_dict should retain the structure and
    naming convensions used in the imas python library.
    """


    def default_config(self):
        config = super().default_config()
        config['emissivity_scale']  = 1.0
        config['temperature_scale'] = 1.0
        config['velocity_scale']    = 1.0
        config['emissivity_file']   = None
        config['temperature_file']  = None
        config['velocity_file']     = None

        config['imas_file'] = None
        config['shot_number'] = None
        config['run_number'] = None

        return config
    
    def initialize(self):
        super().initialize()

        self.data_internal = {}
        interp_psi_2d, r_rho_min, z_rho_min, psi_center, psi_edge = self.initialize_imas()

        self.param['psi_interp']    = interp_psi_2d
        self.param['r_rho_min']     = r_rho_min
        self.param['z_rho_min']     = z_rho_min
        self.param['psi_center']     = psi_center
        self.param['psi_edge']     = psi_edge

        a_interp = self.build_interp_for_plotting()

        self.param['a_interp'] = a_interp

    def initialize_imas(self, wout=None):

        #Build data dictionary from imas database
        data_dict = self.get_imas_data()
        
        #create arrays of data drom the data dictionary
        r2d_data, z2d_data, psi2d_data, psi1d_data, rho1d_equil_data, rho1d_core_data, ionTemp1D_data, emissivity_data = self.load_data_from_imas_dict(data_dict)
       
        #build the needed 1d interpolations for the various quantenties effecting the x-ray generation
        self.build_1d_interpolators(psi1d_data,rho1d_equil_data, rho1d_core_data,ionTemp1D_data,emissivity_data)
        
        #build the 2-d interpolation of the magnetic equilibrium 
        interp_psi_2d = self.build_psi_2d(psi1d_data,psi2d_data,r2d_data,z2d_data)

        r_rho_min  = data_dict['psi_axis_r']
        z_rho_min  = data_dict['psi_axis_z']
        
        #psi_center = data_dict['psi_axis']
        #psi_edge   = data_dict['psi_boundary']

        # There is currently an inconsistency between psi_axis and the
        # 2d psi_data (the 1d psi_data is consistent with psi_axis but not
        # the 2d psi_data). The value of psi_axis is too small, meaning that it
        # cannot be used for s or rho normalization. Until this is resolved
        # we need to take the value of psi_axis from the 2d grid. This is an
        # imperfect solution, but will work reasonably well for most raytracing
        # purposes. This inconsistency is found for shot 131038 (which is the
        # only one we have looked at so far).
        mask = (z2d_data < 4.0) & (z2d_data > -3.5)
        mask[mask] &= (psi2d_data[mask] > np.min(psi1d_data))
        psi_center = np.max(psi2d_data[mask])
        psi_edge = np.min(psi1d_data)

        return interp_psi_2d, r_rho_min,z_rho_min,psi_center,psi_edge

    def get_imas_data(self):

        if self.param['imas_file']:
            data = self.get_imas_data_from_file(self.param['imas_file'])
        else:
            data = self.get_imas_data_from_db(self.param['shot_number'], self.param['run_number'])

        return data

    def build_psi_2d(self,psi1d,psi2d,r2d,z2d):

        #build 2d interpolation function
        interp_psi2d = interpolate.RectBivariateSpline(r2d.T[0],z2d[0],psi2d)
    
        return interp_psi2d

    def load_data_from_imas_dict(self, data_dict):
        
        #unpack data dictionary
        r2d_data = data_dict['R_2d_data'] 
        z2d_data = data_dict['Z_2d_data']
        psi2d_data = data_dict['psi_2d_data']  
        psi1d_data = data_dict['psi_1d_data'] 
        rho1d_equil_data = data_dict['equil_rho_1d_data'] 
        rho1d_core_data = data_dict['core_rho_1d_data'] 
        ionTemp1D_data = data_dict['temp_1d_data']  
        emissivity_data = data_dict['emis_1d_data'] 
        
        return r2d_data, z2d_data, psi2d_data, psi1d_data, rho1d_equil_data, rho1d_core_data, ionTemp1D_data, emissivity_data
    
    def build_1d_interpolators(self, psi1d_data,rho1d_equil_data,rho1d_core_data,ionTemp1D_data,emissivity_data):

        if self.param['temperature_file']:
            self.log.debug(f"Reading temperature profile from: {self.param['temperature_file']}")
            data =  np.loadtxt(self.param['temperature_file'], dtype = np.float64)
        else:
            data = np.array([rho1d_core_data, ionTemp1D_data*1e3]).T
        interp_temp_1D = interpolate.interp1d(
            data[:,0],
            data[:,1],
            fill_value=(np.max(data[:,1]),0),
            bounds_error=False)
        self.data_internal['interp_temp'] = interp_temp_1D

        interp_veloc_1D = None
        self.data_internal['interp_velo'] = interp_veloc_1D

        if self.param['emissivity_file']:
            self.log.debug(f"Reading emissivity profile from: {self.param['emissivity_file']}")
            data =  np.loadtxt(self.param['emissivity_file'], dtype = np.float64)
        else:
            raise NotImplementedError(
                "Impurity charge state emissivities from IMAS not currently supported. "
                "Please use the 'emissivity_file' config option")
        interp_emis_1D = interpolate.interp1d(
            data[:,0],
            data[:,1],
            fill_value='extrapolate',
            bounds_error=False)
        self.data_internal['interp_emis'] = interp_emis_1D

           
    def RZ_from_car(self,car_array):
        'Convert the cartesian coordinates of an nx3 array to cylindrical R,Z used by equilibrium interpolation'
   
        RZ = np.zeros((len(car_array),2))
        RZ[:,0] = np.sqrt(car_array[:,0]**2 + car_array[:,1]**2)
        RZ[:,1] = car_array[:,2]
        
        return RZ
    
    def rho_from_RZ(self,RZ_array,psi_2d_interp_function,psi_center,psi_edge):

        psi_array = psi_2d_interp_function(RZ_array[:,0],RZ_array[:,1],grid=False)
        s = (psi_array - psi_center)/(psi_edge - psi_center)
        s[s < 0.0] = 0.0
        rho_array = np.sqrt(s)

        return rho_array
    
    def rho_from_car(self,car_array,psi_2d_interp_function,psi_center,psi_edge):
    
        RZ_array = self.RZ_from_car(car_array)
        rho_array = self.rho_from_RZ(RZ_array,psi_2d_interp_function,psi_center,psi_edge)
        
        #remove anything outisde of core region
        rho_array[rho_array > 1.0] = np.nan
        rho_array[car_array[:,2] < -3.5] = np.nan
        rho_array[car_array[:,2] > 4.0] = np.nan
        
        return rho_array

    def get_temperature(self, rho):

        interp = self.data_internal['interp_temp']
        output = interp(rho)
        
        return output

    def get_emissivity(self, rho):

        interp = self.data_internal['interp_emis']
        output = interp(rho)
        
        return output

    def get_velocity(self, rho):
        # For now just set velocity to zero.
        output = np.zeros((len(rho),3))
        
        return output
    
    def car_from_flx(self, flx_coord):
        
        a_interp = self.param['a_interp']
        
        r_rho_min = self.param['r_rho_min']
        z_rho_min = self.param['z_rho_min']
        
        rho = np.sqrt(flx_coord[0])
        m = flx_coord[1]
        n = flx_coord[2]
        
        calc_a = a_interp(rho,m)
    
        
        car_point = np.array([[0.0,0.0,0.0]])
        car_point[0][0] = np.cos(n)*(r_rho_min + np.cos(m)*calc_a)
        car_point[0][1] = np.sin(n)*(r_rho_min + np.cos(m)*calc_a)
        car_point[0][2] = z_rho_min + np.sin(m)*calc_a
        
            
        return car_point[0]
    
    def build_interp_for_plotting(self):

        psi_interp = self.param['psi_interp']
        r_rho_min  = self.param['r_rho_min']
        z_rho_min  = self.param['z_rho_min']
        psi_center = self.param['psi_center']
        psi_edge   = self.param['psi_edge']
        
        phi_array = np.linspace(0,2*np.pi,52)        
        output_phi = []
        output_rho = []
        output_a = []
        

        for j in range(len(phi_array)):

            R_start = r_rho_min
            Z_start = z_rho_min
            
            R = R_start
            Z = Z_start
            rho_eval = 0.0

            while rho_eval < 1.0:
                
                #step size
                step_size = .01
        
                #step
                step_R = step_size * np.cos(phi_array[j])
                step_Z = step_size * np.sin(phi_array[j])
        
                R += step_R
                Z += step_Z

                #update rho_eval
                s_eval = (psi_interp(R,Z)[0][0] - psi_center)/(psi_edge - psi_center)
                if s_eval < 0:
                    s_eval = 0.0
                rho_eval = np.sqrt(s_eval)
                
                output_rho.append(rho_eval)
                output_phi.append(phi_array[j])
                output_a.append(np.sqrt((R - R_start)**2+(Z-Z_start)**2))

        a_from_rho_and_phi = interpolate.CloughTocher2DInterpolator(list(zip(output_rho,output_phi)),output_a)
            
        return a_from_rho_and_phi
    
    def bundle_generate(self, bundle_input):

        bundle_input = super().bundle_generate(bundle_input)

        psi_interp   = self.param['psi_interp']
        psi_center   = self.param['psi_center']
        psi_edge     =  self.param['psi_edge']     
        
        profiler.start("Bundle Input Generation")
        m = bundle_input['mask']

        profiler.start("Fluxspace from Realspace")
        rho = self.rho_from_car(bundle_input['origin'][m],psi_interp,psi_center,psi_edge)
        profiler.stop("Fluxspace from Realspace")
        
        
        # evaluate emissivity, temperature and velocity at each bundle location.
        bundle_input['temperature'][m] = self.get_temperature(rho) * self.param['temperature_scale']
        bundle_input['emissivity'][m]  = self.get_emissivity(rho)  * self.param['emissivity_scale']
        bundle_input['velocity'][m]    = self.get_velocity(rho)   * self.param['velocity_scale']
        
        
        fintest = np.isfinite(bundle_input['temperature'])
        m &= fintest
        
        profiler.stop("Bundle Input Generation")

        return bundle_input

    @staticmethod
    def get_imas_data_from_db(shot, run):
        """
        Load data from the ITER IMAS database and return a dictionary.
        This static method needs to be run on the ITER HPC cluster.
        """
        try:
            import imas
        except ImportError:
            logging.error(
                "Could not import the imas module. Are you running on the ITER"
                "hpc cluster with the IMAS module loaded?")
            raise

        # Set initial variables
        user_or_path = 'public'
        database = 'iterdb'
        time = -99
        occ = 0
        it = 0
        version = os.getenv('IMAS_VERSION')[0]
        DB_ABS_PATH = '/work/imas/shared/iterdb/3/0'

        # Get imas shot/run. err,n call seems to be  needed in order to work, not sure why???
        input = imas.ids(shot, run)
        [err, n] = input.open_env(user_or_path, database, version, silent=True)

        # get equilibrium and equilibrium time data
        equilibrium = input.equilibrium
        equilibrium.time = input.equilibrium.partialGet('time', occ)

        # Load the appropriate timeslice.
        equilibrium.time_slice.resize(1)
        equilibrium.time_slice[0] = input.equilibrium.partialGet('time_slice(' + str(it) + ')', occ)

        # 1d equilibrium profiles
        psi1d = equilibrium.time_slice[0].profiles_1d.psi
        equil_rho_1d = equilibrium.time_slice[0].profiles_1d.rho_tor_norm

        psi_axis = equilibrium.time_slice[0].global_quantities.psi_axis
        psi_axis_r = equilibrium.time_slice[0].global_quantities.magnetic_axis.r
        psi_axis_z = equilibrium.time_slice[0].global_quantities.magnetic_axis.z
        psi_boundary = equilibrium.time_slice[0].global_quantities.psi_boundary

        # Cartesian (R,Z) grids
        r2d = equilibrium.time_slice[0].profiles_2d[0].r
        z2d = equilibrium.time_slice[0].profiles_2d[0].z
        psi2d = equilibrium.time_slice[0].profiles_2d[0].psi

        # Get core info
        core_profiles = input.core_profiles
        core_profiles.time = input.core_profiles.partialGet('time')

        # reformat core profile data
        core_profiles.profiles_1d.resize(1)
        core_profiles.profiles_1d[0] = input.core_profiles.partialGet('profiles_1d(' + str(it) + ')')

        # get rho info
        nrho = len(core_profiles.profiles_1d[0].grid.rho_tor_norm)
        core_rho_tor_norm = core_profiles.profiles_1d[0].grid.rho_tor_norm

        # Get profile data
        ion_temperature = core_profiles.profiles_1d[0].t_i_average * 1.e-3
        temp_1d_data = np.array(ion_temperature)

        data_dict = {}

        data_dict['equil_rho_1d_data'] = equil_rho_1d
        data_dict['psi_1d_data'] = psi1d
        data_dict['psi_2d_data'] = psi2d
        data_dict['R_2d_data'] = r2d
        data_dict['Z_2d_data'] = z2d

        data_dict['psi_axis'] = psi_axis
        data_dict['psi_axis_r'] = psi_axis_r
        data_dict['psi_axis_z'] = psi_axis_z
        data_dict['psi_boundary'] = psi_boundary

        data_dict['core_rho_1d_data'] = core_rho_tor_norm
        data_dict['temp_1d_data'] = temp_1d_data
        data_dict['veloc_1d_data'] = None

        return data_dict

    @staticmethod
    def get_imas_data_from_file(filename):
        """
        Load IMAS data from a savefile.
        """
        logging.debug(f"Reading imas data from savefile: {filename}")
        data = xicsrt_io._dict_from_file(filename)

        return data
