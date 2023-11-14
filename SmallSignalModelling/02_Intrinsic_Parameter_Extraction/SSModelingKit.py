import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from typing import List, Tuple
#import logging 
import skrf as rf

__all__ = ['ParasiticAdmittance', 'ParasiticImpedance', 'IntrinsicAdmittance', 'ModeledAdmittance']


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rc('axes',  titlesize = 20)
plt.rc('axes',  labelsize = 20)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)


"""
RF Parameter Modeling, Extraction, and Optimization Library
-----------------------------------------------------------

This library offers a comprehensive suite of classes for modeling, extracting, and optimizing RF S-parameters, 
considering both intrinsic and parasitic components within high-frequency semiconductor devices.

Classes:
- Sparameters: Models and returns complete S-parameters of a transistor, integrating both intrinsic elements and 
  parasitic effects, with methods to provide a scikit-rf Network representation.

- ModeledAdmittance: Calculates intrinsic admittance parameters for small-signal models of transistors based on 
  provided circuit parameters such as capacitances, resistances, and transconductance.

- DirectExtraction: Follows the extraction methodology described by A. Miras for deriving intrinsic parameters 
  directly from S-parameters, including curve-fitting to Y-parameters for parameter extraction.

- ParasiticImpedance: Creates a parasitic impedance matrix from given inductances and resistances, and converts 
  this to S, Y, and Z parameter representations, with verification against Keysight ADS.

- ParasiticAdmittance: Generates a parasitic admittance matrix from capacitances and offers methods to convert to 
  S, Y, and Z parameters, also validated against Keysight ADS simulations.

- Optimization: Provides mechanisms for local and global optimization of circuit parameters using various error 
  functions and optimization algorithms. It supports direct and indirect parameter extraction methods for model 
  fitting and enhancement.

Usage:
To utilize these classes, instantiate with appropriate circuit parameters and frequency data. Each class offers 
methods to access different parameter representations (S, Y, Z) and utility functions for further analysis, such 
as parameter plotting, Network object creation, and optimization routines.

Note:
The accuracy of the parameter extraction, modeling, and optimization is contingent on the precision of the input 
parasitic parameters and the validity of the small-signal model assumptions. Detailed instructions for each class 
are accessible through the `help` property of class instances.

For detailed usage of each class, refer to the individual class documentation by accessing the `help` property 
of an instance.
"""



class ParasiticAdmittance(object):
    """
    ParasiticAdmittance Class
    -------------------------
    This class creates a parasitic admittance matrix given the parasitic capacitances (Cpg, Cpd, and Cpgd) and the frequencies of interest. It provides S-parameters, Y-parameters, Z-parameters, and can also return a two-port network representation of the parasitic capacitive network.
    
    The class has been verified against simulations from Keysight Advanced Design System (ADS).
    
    Methods:
    - Y11, Y22, Y21, Y12: Calculate the individual admittance matrix elements.
    - y: Returns the complete parasitic Y-parameter matrix.
    - s: Converts the Y-parameter matrix to S-parameters.
    - z: Converts the Y-parameter matrix to Z-parameters.
    
    Properties:
    - ntw: Returns a scikit-rf Network object representing the parasitic capacitive network.
    
    Usage:
    Instantiate the class with the parasitic capacitances and frequency points:
    
        parasitic_admittance = ParasiticAdmittance(caps, freq)
    
    Access the admittance, impedance, or S-parameter representations using the respective methods:
    
        Y_parameters = parasitic_admittance.y()
        Z_parameters = parasitic_admittance.z()
        S_parameters = parasitic_admittance.s()
        network = parasitic_admittance.ntw
    
    Note:
    The characteristic impedance is set to 50 Ohms by default but can be changed if necessary.
    """
    
    CHARACTERISTIC_IMPEDANCE = 50 # Ohm
    def __init__(self, caps:List[float], freq:np.ndarray):
        self.freq = freq
        self.Cpg, self.Cpd, self.Cpgd    = caps
        self.w     = 2 * np.pi * freq
    
    def Y11(self) -> complex:
        return 1j * self.w * (self.Cpg + self.Cpgd)
    
    def Y22(self) -> complex:
        return 1j * self.w * (self.Cpd + self.Cpgd)
    
    def Y21(self) -> complex: 
        return 1j * self.w * self.Cpgd
    
    def Y12(self) -> complex:
        return 1j * self.w * self.Cpgd
    
    def y(self)   -> np.ndarray : # to denote that it comes from capacitive network c is added to y, so yc. similarily with the following parameters
        yp        = np.array([self.Y11(), self.Y12(), self.Y21(), self.Y22()]).T.reshape((self.freq.shape[0],2,2))
        return yp 
    
    def s(self)   -> np.ndarray :
        return rf.network.y2s(self.yc())
    
    def z(self)   -> np.ndarray :
        return rf.network.y2z(self.yc())
    
    def ntw(self) -> rf.Network:
        return rf.Network(s = self.sc(), 
                          z0 = self.CHARACTERISTIC_IMPEDANCE,
                          f =self.freq, 
                          f_unit='Hz', 
                          name= 'Parasitic-Cap-Network')

    @property
    def help(self):
        print(self.__doc__)         
    
    

class ParasiticImpedance(object):
    """
    ParasiticImpedance Class
    ------------------------
    This class is responsible for creating a parasitic impedance matrix given the parasitic inductances, resistances,
    and the frequencies of interest. It can provide S-parameters, Y-parameters, Z-parameters, and the two-port network
    representation of the parasitic elements.
    
    The class has been verified against Keysight Advanced Design System (ADS) simulations.
    
    Methods:
    - Z11, Z22, Z21, Z12: Calculate the individual impedance matrix elements.
    - z: Returns the complete parasitic Z-parameter matrix.
    - s: Converts the Z-parameter matrix to S-parameters.
    - y: Converts the Z-parameter matrix to Y-parameters.
    
    Properties:
    - ntw: Returns a scikit-rf Network object representing the parasitic inductive network.
    
    Usage:
    Instantiate the class with the parasitic inductances, resistances, and frequency points:
    
        parasitic_impedance = ParasiticImpedance(ind, res, freq)
    
    Access the impedance, admittance, or S-parameter representations using the respective methods:
    
        Z_parameters = parasitic_impedance.z()
        Y_parameters = parasitic_impedance.y()
        S_parameters = parasitic_impedance.s()
        network = parasitic_impedance.ntw
    
    Note:
    The characteristic impedance is set to 50 Ohms by default but can be changed if necessary.
    """
    CHARACTERISTIC_IMPEDANCE = 50 # Ohm
    
    def __init__(self, ind, res, freq):
        self.lg, self.ld, self.ls = ind
        self.rg, self.rd, self.rs = res
        #---------------
        self.freq  = freq
        self.w     = 2 * np.pi * freq
    
    def Z11(self) -> complex:
        return self.rg + self.rs + 1j * self.w * (self.lg + self.ls)
    
    def Z22(self) -> complex:
        return self.rd +self.rs + 1j * self.w * (self.ld + self.ls)
    
    def Z21(self) -> complex: 
        return self.rs          + 1j * self.w * (self.ls)
    
    def Z12(self) -> complex:
        return self.rs          + 1j * self.w * (self.ls)
    
    def z(self) -> np.ndarray: # to denote that it comes from resistive and inductive network _ri is added to z, so zi. similarily with the following parameters
        zp         = np.array([self.Z11(), self.Z12(), self.Z21(), self.Z22()]).T.reshape((self.freq.shape[0],2,2))
        return zp   
    
    def s(self) -> np.ndarray:
        return rf.network.z2s(self.z_ri())
    
    def y(self) -> np.ndarray:
        return rf.network.z2y(self.z_ri())
    
    @property
    def ntw(self) -> rf.Network:
        return rf.Network(s = self.s_ri(), 
                          z0 = self.CHARACTERISTIC_IMPEDANCE, 
                          f =self.freq, 
                          f_unit='Hz', 
                          name= 'Parasitic-Inductive-Network')

    @property
    def help(self):
        print(self.__doc__)         


class IntrinsicAdmittance(ParasiticAdmittance, ParasiticImpedance):
    
    """
    IntrinsicAdmittance Class
    -------------------------
    This class is responsible for de-embedding parasitic effects from measured S-parameters to extract the intrinsic S-parameters, Y-parameters, and create an intrinsic network representation.

    The de-embedding process removes the influence of parasitic admittance and impedance from the measured network, yielding parameters that more closely represent the inherent behavior of the device under test.

    Methods:
    - de_embed():
        Performs the de-embedding process by subtracting the parasitic admittance and impedance from the measured Y-parameters and converting them to intrinsic Y-parameters.

    Properties:
    - intrinsic_y:
        Returns the intrinsic Y-parameters after de-embedding.

    - s:
        Converts intrinsic Y-parameters to S-parameters.

    - ntwi:
        Returns a scikit-rf Network object representing the intrinsic network.

    - y11, y22, y21, y12:
        Provide access to individual elements of the intrinsic Y-parameter matrix.

    Usage:
    Instantiate the class with the parasitic capacitances, inductances, resistances, and the measured scikit-rf Network object:

        intrinsic_admittance = IntrinsicAdmittance(caps, ind, res, ntw)

    After instantiation, access the intrinsic properties and methods to analyze the de-embedded network:

        Y_parameters = intrinsic_admittance.intrinsic_y
        S_parameters = intrinsic_admittance.s
        intrinsic_network = intrinsic_admittance.ntwi

    Note:
    Ensure the parasitic parameters provided are accurate to achieve a reliable de-embedding process. The results are highly dependent on the quality of the parasitic parameter models.
    """
    def __init__(self, caps, ind, res, ntw):
        self.ntw           = ntw
        self.y_measured    = self.ntw.y
        self._freq         = ntw.f

        self.parasitic_admittance = ParasiticAdmittance(caps, self._freq).y()
        self.parasitic_impedance  = ParasiticImpedance(ind, res, self._freq).z()
        self._intrinsic_y         = None  # Cache for intrinsic Y-parameters


    @property
    def intrinsic_y(self):
        if self._intrinsic_y is None:
            self._intrinsic_y = self.de_embed()
        return self._intrinsic_y        
    

    def de_embed(self):  # De-embeds the parasitics based on pY and pZ definitions
        y_corrected     = self.y_measured - self.parasitic_admittance
        z_corrected     = rf.network.y2z(y_corrected)
        z_intrinsic     = z_corrected - self.parasitic_impedance
        y_intrinsic     = rf.network.z2y(z_intrinsic) 
        return   y_intrinsic
        
    
    @property
    def s(self):
        return rf.network.y2s(self.intrinsic_y) 
    
    @property
    def ntwi(self):
        return rf.Network(s = self.s, z0=50, f = self._freq, f_unit='Hz', name='intrinsic_network') 
    
    def y11(self):
        return  self.intrinsic_y[:,0,0]
    
    def y22(self):
        return  self.intrinsic_y[:,1,1]
    
    def y21(self):
        return self.intrinsic_y[:,1,0]
    
    def y12(self):
        return self.intrinsic_y[:,1,0]
    
    @property
    def help(self):
        print(self.__doc__)  


  
  

    
        

class DirectExtraction(IntrinsicAdmittance): 
    """
   DirectExtraction Class
   ----------------------
   This class implements a direct extraction method for intrinsic parameters from S-parameters, following the procedure described by A. Miras in "Very High Frequency Small-Signal Equivalent Circuit for Short Gate-Length InP HEMT's", IEEE TMTT, 1997.

   The class focuses on two main approaches for intrinsic parameter extraction:
   1. Direct solving of Y-parameter equations for intrinsic parameters.
   2. Curve-fitting to Y-parameters to extract intrinsic parameters, as followed in the reference paper.

   The class assumes Rgd (gate-drain resistance) to be infinite.

   It has been tested against ADS-generated S-parameters with excellent results, except for the tau parameter. For measured S-parameters of real devices, the extraction of Ri (input resistance), Rj (junction resistance), and tau (time delay) can produce unrealistic results.

   Methods:
   - D1(): Calculates the first denominator term used in intrinsic admittance calculations.
   - D2(): Calculates the second denominator term used in intrinsic admittance calculations.
   - polFit(x, y): Fits a polynomial curve of order 2 to the provided data.
   
   Intrinsic Parameters Extraction:
   - cgd(): Extracts the gate-drain capacitance.
   - cgs(): Extracts the gate-source capacitance.
   - cds(): Extracts the drain-source capacitance.
   - Rj(): Extracts the  gate-drain resistance.
   - Rgs(): Extracts the gate-source resistance based on low-frequency real part of Y11 + Y12.
   - Ri(): Extracts the input resistance.
   - Rds(): Extracts the drain-source resistance.
   - gds(): Extracts the drain-source conductance.
   - gm(): Extracts the transconductance.
   - tau(): Extracts the time delay, which may need further exploration due to potential unrealistic values.

   - ipm(): Returns the average of intrinsic parameters, which can serve as a starting point for optimization.
   - plot_parameters(pm, labels, name): Plots the given parameters against frequency with provided labels and a name.
   - plot_conductances(): Plots conductance parameters.
   - plot_capacitances(): Plots capacitance parameters.
   - plot_resistances(): Plots resistance parameters.

   Properties:
   - ntw: Returns a scikit-rf Network object representing the modeled S-parameters.

   Usage:
   Instantiate the class with a Network object and the parasitic parameters, then call the methods to extract intrinsic parameters:

       direct_extraction_instance = DirectExtraction(ntw, p_capacitance, p_inductance, p_resistance)
       intrinsic_parameters = direct_extraction_instance.ipm()
   
   Note:
   This class is designed to be used with high-frequency small-signal models, particularly for Short Gate-Length InP HEMT devices.
    """
    LOW_FREQUENCY_LIMIT = 47 # 
    POLYNOMIAL_ORDER    = 2
    def __init__(self, ntw, p_capacitance, p_inductance, p_resistance):
        self.ntw  =  ntw
        self.freq = ntw.f 
        self.w    = 2*np.pi*self.freq
        self.y    = IntrinsicAdmittance(p_capacitance, p_inductance, p_resistance, self.ntw).intrinsic_y 
        
        self._D1  = None 
        self._D2  = None 
    
    
    def D1(self):
        if self._D1 is None:
            wct      = ( self.y[:,0,0].real + self.y[:,0,1].real ) / ( self.y[:,0,0].imag + self.y[:,0,1].imag )
            self._D1 = 1+ ( wct )**2
        return self._D1
    
    def D2(self):
        if self._D2 is None:
            wct      = (  self.y[:,0,1].real ) / (  self.y[:,0,1].imag )
            self._D2 = 1 + ( wct )**2
        return self._D2        
    
    def polFit(self, x, y):
        polCoef      = np.polyfit(x,y,self.POLYNOMIAL_ORDER)
        polF         = np.polyval(polCoef,x)
        return polF
        

    #---------------------------------------------------------------------#
    #--------------------- Intrinsic Parameters --------------------------#
    #---------------------------------------------------------------------#
    def cgd(self):
        return     - self.D2() * self.y[:,0,1].imag / self.w
    
    def cgs(self):
        return      self.D1() * ( self.y[:,0,0].imag + self.y[:,0,1].imag ) / self.w
    
    def cds(self):
        return    ( self.y[:,0,1].imag + self.y[:,1,1].imag) / self.w
    
    def Rj(self):         
        return   self.y[:,0,1].real / (self.y[:,0,1].real**2 + self.y[:,0,1].imag**2)  
    
    def Rgs(self):                                                               # is derived from the low frequency Re(Y11 + Y12), because Re(Y11+Y12) ~ w**2 + 1/ Rgs as w->0,
        x   = self.w[:self.LOW_FREQUENCY_LIMIT]**2                               # a polynomial curve of 2nd order is fit to data upto 5 GHz. Obviously there are large uncertainties depending on fitting
        y   = self.y[:,0,0].real +  self.y[:,0,1].real                           # if included in the SSM it would have to be optimized further.
        rgs = 1 / self.polFit(x,y[0][:self.LOW_FREQUENCY_LIMIT])[0]              # The accuracy of this is debatable due to noise low frequency data; 
        return rgs                                                               # still it can provide a starting point for Rgs that can be further optimized later.
  
    def Ri(self):
        return    (  self.y[:,0,0].real + self.y[:,0,1].real) / (  self.y[:,0,0].imag +  self.y[:,0,1].imag ) / self.cgs() / self.w 
     
    def Rds(self):
        return    1 / (  self.y[:,0,1].real + self.y[:,1,1].real ) 
    
    def gds(self):
        return     self.y[:,0,1].real + self.y[:,1,1].real
    
    def gm(self):
        return      np.sqrt(  self.D1()* (   (self.y[:,1,0].imag - self.y[:,0,1].imag)**2 + ( self.y[:,1,0].real - self.y[:,0,1].real)**2  )  )  
    
    def tau(self): # gives unrealistic values needs to be explored
        return      np.angle( ( self.y[:,1,0] - self.y[:,0,1] ) * ( 1 + 1j * ( self.y[:,0,0].real + self.y[:,0,1].real ) / ( self.y[:,0,0].imag  +  self.y[:,0,1].imag) ) )*1e-12
     
    
    def ipm(self):                                                         # Returns the averages of intrinsic parameters in a list --> This can be used as a starting point for optimization
         avP  = lambda x, m=10,n=len(self.freq): np.mean(x[m:n])
         pm   = [avP(self.cgs()), avP(self.cgd()), avP(self.cds()), avP(self.Ri()), avP(-self.Rj()), avP(self.Rds()), avP(self.gm()), avP(self.tau())]
         return pm
     

        
    def plot_parameters(self, pm, labels, name):
        """
        Plot the given parameters against frequency with provided labels and a name.
        
        """
        fig, ax = plt.subplots(1, figsize=(10,7))
        k=0
        for i, label in enumerate(labels):
            ax.plot(self.freq/1e9, pm[i], label=label)
            k+=1
        
        ax.set_xlabel("Frequency (GHz)", fontsize=18)
        ax.set_ylabel(name, fontsize=18)
        ax.set_xlim(np.min(self.freq/1e9), np.max(self.freq/1e9))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=16)
        fig.tight_layout()
        plt.show()
        
    def plot_conductances(self):
        pm     = [self.gm(), self.gds()] 
        labels = ['gm', 'gds']
        return  self.plot_parameters(pm, labels, name='Conductances (S)')
    
    def plot_capacitances(self):
        pm     =  [self.cgs(), self.cgd(), self.cds()] 
        labels = ['Cgs', 'Cgd', 'Cds']
        return self.plot_parameters(pm, labels, name='Capacitances (F)')
    
    def plot_resistances(self):
        pm     = [self.cgs(), self.cgd(), self.cds()] 
        labels = ['Ri', 'Rj']
        return self.plot_parameters(pm, labels, name='Resistances (Ohm)')  
     
    @property
    def help(self):
        print(self.__doc__)         
    



class ModifiedNetwork:   
    """
    ModifiedNetwork Class
    ---------------------
    This class provides static methods to modify the S-parameters of a given scikit-rf Network object. These modifications can include reordering S-parameters, magnifying or shrinking certain S-parameters, and filtering S-parameters based on a specific frequency range.

    Methods:
    - reorder_sp(ntw):
        Reorders S-parameters such that s11<->s22 and s21<->s12 are swapped.
    
    - reorder_network(ntw):
        Returns a new Network with reordered S-parameters.
    
    - filtered_network(ntw, f_low, f_high, f_step):
        Filters the Network to only include S-parameters within a specified frequency range.
    
    - zoom_ntw(ntw, filtered):
        Magnifies s12 and shrinks s21 by specified factors to make them more visible on a Smith chart.
    
    Parameters:
    - ntw: A scikit-rf Network object whose S-parameters are to be modified.
    - f_low: The lower bound of the frequency range for filtering (in GHz).
    - f_high: The upper bound of the frequency range for filtering (in GHz).
    - f_step: The frequency step for filtering (in GHz).
    - filtered: A boolean flag to determine if filtering should be applied before zooming.
    
    Usage:
    To use these methods, you do not need to instantiate the class since they are static methods. Here is an example:

        ntw = rf.Network('path_to_s2p_file.s2p')
        reordered_ntw = ModifiedNetwork.reorder_network(ntw)
        filtered_ntw = ModifiedNetwork.filtered_network(ntw, f_low=0.1, f_high=18.1, f_step=0.1)
        zoomed_ntw = ModifiedNetwork.zoom_ntw(ntw, filtered=True)
    
    Note:
    - When reordering S-parameters, ensure that the physical setup matches the expected port configuration.
    - The zoom_ntw method is particularly useful for visualizing S-parameters on a Smith chart where certain parameters are too small or too large to be easily seen.
    """
    
    @staticmethod
    def reorder_sp(ntw):
        s11 = ntw.s11.s
        s12 = ntw.s12.s
        s21 = ntw.s21.s
        s22 = ntw.s22.s
        return np.concatenate((s22,s21,s12,s11), axis=2).reshape(ntw.s.shape)
    
    @staticmethod
    def reorder_network(ntw):
        return rf.Network(s=ModifiedNetwork.reorder_sp(ntw), f=ntw.f,  f_unit='Hz', name='S_parameters_Reversed')
    
    @staticmethod
    def filtered_network(ntw, f_low=0.1, f_high=18.1, f_step=0.1):             # used to create a subset S-parameter dataset by removing or filtering unwanted frequency points
        Np    = int((f_high-f_low)/f_step) + 1                                 # Number of frequency points
        freqF = rf.Frequency(f_low, f_high, npoints=Np, unit='ghz').f/1e9      # Defines new frequencies based on scikit rf, this is needed for the scikit rf Network; otherwise simple numpy array would work
        freqC = ntw.f/1e9
        sp    = ntw.s                                                          # initial S-parameters that we want to filter; a susbset of these S-parameters is created here
        
        idx   = np.isin(freqC, freqF)                                          # isolates the indices that correspond to frequency points FreqF inside the array freqC (corresponds to dataset that we want to filter)
        f_sp  = sp[idx]                                                        # keeps only the s-parameters that correspond to idx
        
        sp_filtered         = np.array([f_sp[:,0,0], f_sp[:,0,1], f_sp[:,1,0], f_sp[:,1,1]]).T.reshape((freqF.shape[0],2,2)) 
        return rf.Network(s = sp_filtered, f = freqF*1e9, f_unnit='Hz', name='Sp_filtered')
    
    def zoom_ntw(ntw, filtered = False): 
        """
        # this is used to shrink s21 and magnify s12 ; so that they can be seen on smith chart.
        
        """
        ntwF  = ntw 
        freqF = ntw.f 
        if filtered: 
            ntwF  = ModifiedNetwork.filtered_network(ntw)
            freqF = ntwF.f 
            
        s11      = ntwF.s11.s
        s12      = ntwF.s12.s * 2.5  # Magnify S12 2.5 times
        s21      = ntwF.s21.s / 10   # shrink s21 by 10
        s22      = ntwF.s22.s
        ##################
        SpM      = np.array([s11,s12,s21,s22]).T.reshape(ntwF.s.shape)
        return rf.Network(s = SpM, f=freqF, f_unit = 'Hz', name='Sp_magnified_and_shrunk' )
    
    @property
    def help(self):
        print(self.__doc__)





class ModeledAdmittance(object):
    """
    ModeledAdmittance Class
    -----------------------
    This class creates intrinsic Y-parameters for small-signal models of transistors when given the equivalent circuit parameters.

    Parameters:
    - ipm: List or array of intrinsic parameters in the following order:
        [cgs, cgd, cds, ri, rj, rds, gm, tau]
    - freq: Array of frequency points where the S-parameters are to be evaluated.

    The intrinsic parameters are:
    - cgs: Gate-source capacitance in farads.
    - cgd: Gate-drain capacitance in farads.
    - cds: Drain-source capacitance in farads.
    - ri: Gate resistance in ohms.
    - rj: Drain resistance in ohms.
    - rds: Drain-source resistance in ohms.
    - gm: Transconductance in siemens.
    - tau: Time delay in seconds.

    Methods:
    - Zp: Calculates the total Z-parameters including parasitic effects.
    - Yp: Calculates the total Y-parameters including parasitic effects.
    - y: Returns the intrinsic Y-parameters as a function of frequency.
    - z: Converts the intrinsic Y-parameters to Z-parameters.
    - s: Converts the intrinsic Y-parameters to S-parameters.
    
    Properties:
    - ntw: Returns a scikit-rf Network object representing the intrinsic S-parameters.

    Usage:
    To use this class, instantiate it with the intrinsic parameters and frequency points:

        modeled_admittance = ModeledAdmittance(ipm, freq)

    You can then access the modeled Y, Z, and S-parameters or the corresponding scikit-rf Network object:

        Y_parameters = modeled_admittance.y()
        Z_parameters = modeled_admittance.z()
        S_parameters = modeled_admittance.s()
        network = modeled_admittance.ntw

    Note:
    - The current model does not include Rgs and Rgd, which are present in many models.
    - This may be tested and added later if needed.
    """
    def __init__(self, ipm, freq):  # ipm = [cgs, cgd, cds, ri, rj, rds, gm, tau]
        self.cgs, self.cgd, self.cds, self.ri, self.rj, self.rds, self.gm, self.tau = ipm   
        self.freq = freq 
        self.w    = 2 * np.pi * self.freq
        self._D1  = None 
        self._D2  = None 
        self._y   = None 
    
    
    def D1(self):
        if self._D1 is None :
            self._D1 = 1 + ( self.w * self.cgs*1e-15 * self.ri )**2 
        return self._D1
    
    def D2(self):
        if self._D2 is None:
            self._D2 = 1 + ( self.w * self.cgd*1e-15 * self.rj )**2
        return self._D2


    def calculate_admittances(self):
        if self._D1 or self._D2 is None:
            self.D1()
            self.D2()
            
        if self._y is None:
            y11_r = self.ri * ( self.cgs*1e-15 * self.w ) **2 / self._D1 + self.rj * ( self.cgd*1e-15 * self.w )**2 / self._D2 
            y11_i = self.w  * ( self.cgs*1e-15 / self._D1 + (self.cgd*1e-15) / self._D2 ) 
            #-----#
            y12_r = - self.rj * ( self.cgd*1e-15 * self.w )**2 / self._D2
            y12_i = - self.w  * ( self.cgd*1e-15) / self._D2
            #-----#
            y21_r = - self.rj* ( self.cgd*1e-15 * self.w )**2 / self._D2
            y21_g = (self.gm*1e-3) * np.exp(-1j * self.w * (self.tau*1e-12)) / (1 + 1j* self.w * self.cgs*1e-15 * self.ri)
            y21_i    = - self.w  * (self.cgd*1e-15) / self._D2
            #-----#
            y22_r = 1 / self.rds + self.rj* (self.cgd*1e-15*self.w)**2 / self._D2
            y22_i = self.w * ( self.cds*1e-15 + self.cgd*1e-15 / self._D2 )

            # Create the admittance matrix for all frequencies at once
            y11   = y11_r +1j * y11_i 
            y12   =  y12_r +1j * y12_i 
            y21   = y21_g + y21_r + 1j*y21_i  # Assuming y21 should be the same as y12 with added gm component
            y22   = y22_r + 1j * y22_i
            
            self._y = np.array([y11, y12, y21, y22]).T.reshape((self.freq.shape[0],2,2))
        
        return self._y

    
    def y(self):  # modelled Y-parameters for the intrinsic SSM
        if self._y is None:
            self.calculate_admittances()
        return self._y
    
    def z(self): # modelled Z-parameters for the intrinsic SSM
        if self._y is None:
            self.calculate_admittances()    
        return rf.network.y2z(self._y)
    
    def s(self):  # modelled S-parameters for the intrinsic SSM
        if self._y is None:
            self.calculate_admittances()
        return rf.network.y2s(self._y)
    
    @property
    def ntw(self): # modelled rf scikit network for the intrinsic SSM
        return rf.Network(s=self.s(), f=self.freq, f_unit='Hz', name='Intrinsic_network')
    
    @property
    def help(self):
        print(self.__doc__)
    
        

class Sparameters(ParasiticAdmittance, ParasiticImpedance, ModeledAdmittance): 
    """
    Sparameters Class
    -----------------
    This class models the S-parameters of the transistor including parasitic effects and returns S-parameters in array format and as a Network (ntw) object from the scikit-rf package.
    
    Methods
    -------
    Zp():
        Adds the effect of parasitic resistive and inductive elements to the intrinsic Z-parameters.
    
    Yp():
        Adds the effect of parasitic capacitances to the (intrinsic + parasitic resistive/inductive) Y-parameters.
    
    Properties
    ----------
    sp:
        Returns the S-parameters of the transistor containing all parasitics and intrinsic elements.
    
    ntw:
        Returns a scikit-rf Network object representing the modeled S-parameters.
    
    Parameters
    ----------
    caps : list or array
        Parasitic capacitances.
    
    ind : list or array
        Parasitic inductances.
    
    res : list or array
        Parasitic resistances.
    
    ipm : list or array
        Initial parameter values for the optimization.
    
    freq : array
        Frequency points used for S-parameter measurements.
    """
    def __init__(self, caps, ind, res, ipm, freq):
        self.freq         = freq 
        self.parasitic_y  = ParasiticAdmittance(caps, self.freq)
        self.parasitic_z  = ParasiticImpedance(ind, res, self.freq)
        self.modeled      = ModeledAdmittance(ipm, self.freq)
   

    def Zp(self):                                                              # Adds the effect of parasitic resistive and inductive elements to the intrinsic Z-parameters
        return self.modeled.z() + self.parasitic_z.z() 
    
    def Yp(self):                                                              # Adds the effect of parasitic capacitances to the ( intrinsic + parasitic resistive/inductive) Y-parameters
        yp = rf.network.z2y(self.Zp())
        return  yp + self.parasitic_y.y()
    @property
    def sp(self):
        return rf.network.y2s(self.Yp())  
                                     # S-parameters of the transistor containing all parasitics and intrinsic elements
    @property
    def ntw(self):
        return rf.Network( s = self.sp, f = self.freq, f_unit='Hz', name=' Modelled_S_Parameters')
    
    @property
    def help(self):
        print(self.__doc__)
    
        



class Optimization(Sparameters) :
    """
        Optimization Class
        ==================

        This class is designed to optimize the intrinsic parameters of small-signal models for HEMTs.
 
        ## Uses global and local optimizers to optimize intrinsic parameters
        ## The optimizers are taken from scipy and are described here:https://docs.scipy.org/doc/scipy/reference/optimize.html
        
        Methods
        -------
        optimize(method='SLSQP', error_type='nmse'):
            Perform optimization using the selected algorithm and error function.
            Available methods: 'SLSQP', 'Nelder-Mead', 'Powell', 'CG', 'TNC', 'SA'.
            Available error types: 'nme','nmse', 'nmae', 'nrmse', 'relative_error'.
        
        modeled_sp(params):
            Calculate the modeled S-parameters based on the given intrinsic parameters.
        
        total_error(params):
            Calculate the total error of the model compared to the measured S-parameters.
        
        callback(params):
            A callback function that can be used to track the progress of the optimization.

        Attributes
        ----------
        BOUNDS:
            A list of tuples specifying the bounds for the optimization parameters.
        measured_sp:
            Measured S-parameters.
        freq:
            Frequency points used for S-parameter measurements.
        _ipm:
            Initial parameter values for the optimization.

        Usage
        -----
        To use this class, instantiate it with the initial parameters and then call the 'optimize' method:
        
            optim_instance = Optimization(caps, ind, res, ipm, ntw)
            result = optim_instance.optimize(method='SLSQP', error_type='nmse')
        
        After optimization, you can access the optimized parameters with 'optim_instance._ipm'.
    """
    BOUNDS  = [(0,200), (0, 200), (0, 200), (0.1, 10), (0.1, 10), (0, 200), (0,300), (0.1, 2.0) ] # bounds set based on experience with InP, GaAs and GaN low noise HEMTs
    #            Cgs      Cgd       Cds         Ri         Rj       Rds       gm         tau      # we have rarely seen values outside of these limits. 
    #            /fF      /fF       /fF         /Ohm       /Ohm     /Ohm      /mS        /ps 
    
    
    def __init__(self, caps, ind, res, ipm, ntw):
        self.measured_sp = ntw.s                                                # measured S-parameters in (#freq_points, 2,2) array format
        self.freq        = ntw.f                                                # frequencies points used for S-param measurements
        self._ipm        = ipm
        self.N           = self.freq.shape[0]                                   # Number of frequency points
        self._iter       = []
        self._eVal       = []
        self.caps        = caps 
        self.ind         = ind 
        self.res         = res 

                                                     
 
    def modeled_sp(self, params):
        return Sparameters(self.caps, self.ind, self.res, params, self.freq).sp
     
    """ Normalized Squared Deviation (NSD) -> PhD Thesis Mikael Malkvist """   
    def _nsd(self, params):  
        modeled_sp   = self.modeled_sp(params)
        de           = np.mean( np.sum( np.abs(self.measured_sp - modeled_sp)**2, axis=0) / np.max( np.abs(self.measured_sp)**2, axis=0) ) 
        return  de  / self.N 
    
    """ Normalized Composite Deviation (NCD) -> Parameter extraction and complex nonlinear transistor models -> Kompa """   
    def _ncd(self, params):  
        modeled_sp        = self.modeled_sp(params)
        diff_real         = (self.measured_sp - modeled_sp).real 
        diff_imag         = (self.measured_sp - modeled_sp).imag 
        norm              = np.mean( np.sum( np.abs(diff_real ) + np.abs(diff_imag), axis=0) / np.max( np.abs(self.measured_sp), axis=0) ) 
        return  norm  / self.N 
    
    
    """ ormalized Mean Squared Deviation (NMSD) """
    def _nmse(self,params):
        modeled_sp   = self.modeled_sp(params)
        mae          = np.sum( np.abs(self.measured_sp- modeled_sp)**2, axis=0)
        norm         = np.sum( np.abs(self.measured_sp)**2, axis=0)
        return np.mean(mae/norm) if np.all(norm) else float('inf')
    
    """ Normalized Mean Absolute Deviation (NMAD) """
    def _nmae(self,params):
        modeled_sp   = self.modeled_sp(params)
        mae          = np.sum( np.abs(self.measured_sp- modeled_sp), axis=0)
        norm         = np.sum( np.abs(self.measured_sp), axis=0)
        return np.mean(mae/norm) if np.all(norm) else float('inf')
        
    """ Nomrmalized Root Mean Square Error (NMSE) """
    def _nrmse(self,params):
        modeled_sp   = self.modeled_sp(params)
        rmse         = np.sqrt( np.sum( np.abs(self.measured_sp - modeled_sp)**2, axis=0))
        ms_range     = np.max( np.abs(self.measured_sp)**2, axis=0)-np.min(np.abs(self.measured_sp)**2,axis=0) 
        return np.mean(rmse / ms_range) if ms_range !=0 else float('inf')
    
    """ Relative Error """
    def _relative_error(self,params):
        modeled_sp   = self.modeled_sp(params) 
        rel_error    = np.abs( self.measured_sp - modeled_sp) / np.abs(self.measured_sp)
        return np.mean(rel_error) / self.N  if self.N and np.all(self.measured_sp) else float('inf')
    

    
    #######################################################################
    ##################### Optimization            #########################
    #######################################################################
    
    
    def optimize_local(self, method='SLSQP', error_type='nmse'):
        
        # Access the error function based on error_type
        error_function = getattr(self, f'_{error_type}')
        
        # Define a local callback that captures the error_function
        def local_callback(params):
            self._iter.append(self._iter[-1] + 1 if self._iter else 1)
            error_val = error_function(params)
            self._eVal.append(error_val)
        
        options     = {'maxfev': 10000} if method in ['Nelder-Mead', 'Powell', 'CG'] else {}
        result      = optimize.minimize(error_function, self._ipm, method=method, bounds=self.BOUNDS, callback=local_callback, options=options)
        if result.success:
            self._ipm = result.x
            self.print_results(result)
        return result.x
        
    
    def optimize_global(self, method='dual_annealing', error_type='nmse'):
        # Access the error function based on error_type
        error_function = getattr(self, f'_{error_type}')     

        # Define a local callback that captures the error_function
        def local_callback(x, f, context):
            self._iter.append(self._iter[-1] + 1 if self._iter else 1)
            self._eVal.append(f)   
        
        # Choose the global optimization method
        if method   == 'dual_annealing':
            result   = optimize.dual_annealing(error_function, bounds=self.BOUNDS, callback=local_callback)
        elif method == 'differential_evolution':
            result   = optimize.differential_evolution(error_function, bounds=self.BOUNDS, callback=local_callback)
        elif method == 'basinhopping':
            result   = optimize.basinhopping(error_function, self._ipm, niter=100, minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': self.BOUNDS})

                # Update the internal parameter values with the optimization result
        if result.success:
            self._ipm = result.x
            self.print_results(result)
        
        return result.x
    
    #######################################################################
    #######################################################################

    def eVal(self):
        return self._eVal
    
    def iteR(self):
        return self._it
              
        
    def print_results(self, result):
        if result.success:
            print(f"Optimized parameter values are:\n{result.x}")
            print(f"Error Function Value at Optimized values:\n{np.round(result.fun*100,2)} %")
            #self._iter = np.arange(0, self._it)
            self.plot_errorF()
        else:
            print(f"Optimization failed with message: {result.message}") 
        
    def plot_errorF(self):   
        if self._iter and self.eVal:
            # plots the error function values Vs. iterations
            plt.figure(figsize=(7,5))
            plt.plot(self._iter, np.array(self._eVal)*100, c='k')
            plt.xlabel("# Iterations", fontsize=18)
            plt.ylabel(" Error Function Value (%)", fontsize=18)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.show()

    @property
    def help(self):
        print(self.__doc__)
        
    
    
    
        


        












































