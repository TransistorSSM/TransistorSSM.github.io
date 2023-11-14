import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from typing import List, Tuple
#import logging 
import skrf as rf






class Noise:
    """
    Implements the Pospieszalski noise model for calculating noise parameters from a Small Signal Model (SSM) parameters.
    This class includes calculations for minimum noise temperature (Tmin), optimum noise impedance (Zopt), 
    Noise Resistance (Rn), Noise Conductance (gn), and models the Noise Temperature of SSM (of FETs) at a generator impedance Zg.

    Attributes:
        To (float): Standard temperature, set to 290K.
        Zo (float): Characteristic impedance, set to 50 ohms.

    Methods:
        f_T(form='intrinsic'): 
            Calculates the unity current gain cut-off frequency. 
            Params:
                form (str): The form for calculation ('intrinsic' or 'extrinsic').
            Returns:
                float: Unity current gain cut-off frequency.

        Xopt(): 
            Calculates the optimum reactance.
            Returns:
                float: Optimum reactance value.

        Ropt(Tg, Td): 
            Calculates the optimum resistance.
            Params:
                Tg (float): Generator temperature in Kelvin.
                Td (float): Drain temperature in Kelvin.
            Returns:
                float: Optimum resistance value.

        Rn(Tg, Td): 
            Calculates the noise resistance.
            Params:
                Tg (float): Generator temperature in Kelvin.
                Td (float): Drain temperature in Kelvin.
            Returns:
                float: Noise resistance.

        Tmin(Tg, Td): 
            Calculates the minimum noise temperature.
            Params:
                Tg (float): Generator temperature in Kelvin.
                Td (float): Drain temperature in Kelvin.
            Returns:
                float: Minimum noise temperature.

        gn(Td): 
            Calculates the noise conductance.
            Params:
                Td (float): Drain temperature in Kelvin.
            Returns:
                float: Noise conductance.

        N(Tg, Td): 
            Calculates N = Ropt * gn.
            Params:
                Tg (float): Generator temperature in Kelvin.
                Td (float): Drain temperature in Kelvin.
            Returns:
                float: N value.

        Zopt(Tg, Td): 
            Calculates the optimum impedance.
            Params:
                Tg (float): Generator temperature in Kelvin.
                Td (float): Drain temperature in Kelvin.
            Returns:
                complex: Optimum impedance.

        Gamma_opt(Tg, Td): 
            Calculates the optimum reflection coefficient.
            Params:
                Tg (float): Generator temperature in Kelvin.
                Td (float): Drain temperature in Kelvin.
            Returns:
                complex: Optimum reflection coefficient.

        Gamma_g(Zg): 
            Calculates the reflection coefficient of the generator.
            Params:
                Zg (complex): Generator impedance.
            Returns:
                complex: Reflection coefficient of the generator.

        Tn(Tg, Td, Zg): 
            Calculates the noise temperature at a generator impedance Zg.
            Params:
                Tg (float): Generator temperature in Kelvin.
                Td (float): Drain temperature in Kelvin.
                Zg (complex): Generator impedance.
            Returns:
                float: Noise temperature.
    """
    
    To                                = 290  # K
    Zo                                = 50   # Ohm  
    def __init__(self, parasitic_caps, parasitic_res, intrinsic_pm, freq):
        self.cpg, self.cpd, self.cpgd = parasitic_caps 
        self.rg, self.rd, self.rs     = parasitic_res 
        self.cgs, self.cgd, self.cds, self.ri, self.rj, self.rds, self.gm, self.tau = intrinsic_pm 
        self.rgs                      = self.rg + self.ri + self.rs
        self.gds                      = 1 / self.rds if self.rds != 0 else float("inf")
        self.freq                     = freq 
        self.w                        = 2 * np.pi * self.freq
        self._fT                      = None

        
       
    
    def f_T(self, form='intrinsic'):
        
        if form == 'intrinsic':
            total_cap = self.cpg + self.cgs + self.cgd +self.cpd + self.cpgd
        elif form == 'intrinsic':
            total_cap =  self.cgs + self.cgd 
        else:
            raise ValueError("Invalid value for 'form'. Expected 'intrinsic' or 'extrinsic'.")
         
        return self.gm / 2 / np.pi / ( ( total_cap ) * (1 + self.gds*self.rs) + self.cgs*self.gm**self.rs )
            
    def Xopt(self):
        return 1 / (self.cgs * self.w) if np.all( self.cgs*self.w ) !=0 else float('inf')
    
    
    def Ropt(self, Tg, Td):
        if self._fT is None:
            self._fT = self.f_T()
            
        return np.sqrt( (self._fT**2 / self.freq**2) * self.rgs * Tg/ Td + self.rgs**2)
    
    
    def Rn(self, Tg, Td):
        return Tg / self.To * self.rgs + (Td / self.To) * (self.gds / self.gm**2)*( 1 + ( self.w* self.cgs * self.rgs )**2 )
    
    def Tmin(self, Tg, Td):
        if self._fT is None:
            self._fT = self.f_T()
            
        return 2* (self.freq/self._fT) *np.sqrt( self.gds*self.rgs*Td*Tg + (self.freq * self.rgs * self.gds * Td / self._fT )**2 ) + 2*(self.freq**2/self._fT**2)*self.rgs*self.gds*Td
    
    def gn(self, Td):
        if self._fT is None:
            self._fT = self.f_T()
            
        return (self.freq**2 / self._fT**2)*self.gds*Td/self.To 
    
    
    def N(self, Tg, Td):
        return self.Ropt(Tg,Td)*self.gn(Td)
    
    def Zopt(self,Tg,Td):
        return self.Ropt(Tg,Td) + 1j*self.Xopt()
    
    def Gamma_opt(self, Tg, Td):
        return ( self.Zopt(Tg,Td) - self._Zo ) / ( self.Zopt(Tg,Td) + self.Zo )
    
    def Gamma_g(self, Zg):
        return (Zg - self.Zo) / (Zg + self.Zo)
    
    def Tn(self, Tg, Td, Zg):
        gamma_opt = self.Gamma_opt(Tg,Td)
        gamma_g   = self.Gamma_g(Zg)
        Msm       = np.abs( gamma_g - gamma_opt )**2 / ( 1 - np.abs( gamma_opt )**2 ) / ( 1 - np.abs( gamma_g  )**2 )
        return self.Tmin(Tg,Td) + 4*self.N(Tg,Td)*self.To*Msm
    
    
    @property
    def help(self):
        print(self.__doc__)    
    
        
        
    
    
    
