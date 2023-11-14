import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skrf import media
import skrf as rf
from skrf import Network,network


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rc('axes',  titlesize = 20)
plt.rc('axes',  labelsize = 20)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)

class SParamPlot:
    """
    The SParamPlot class provides convenient methods to visualize S-parameters from one or multiple scikit-rf Network objects. 
    It enables easy generation of both Smith charts and magnitude (dB) plots for various S-parameters. 
    
    The class can handle single Network objects or lists of Network objects, allowing for comparative plotting of multiple datasets.
    
    Parameters:
    ntw (rf.Network or list of rf.Network): A single Network object or a list of Network objects to be plotted.
    name (str or list of str, optional): Custom names for each Network object. Defaults to 'Network {i}'.
    
    Methods:
    plot_smith(m, n, title=None):
        Plots the specified S-parameter (m, n) on a Smith chart for each Network in the list.
        
    plot_db(m, n, title=None):
        Plots the specified S-parameter (m, n) in dB scale for each Network in the list.
        
    plot_all_s_smith():
        Plots all four S-parameters on the same Smith chart for visual comparison.
        
    plot_all_s_db():
        Plots all four S-parameters in dB scale in a 2x2 subplot arrangement for each Network.
        
    plot_combined_network_parameters():
        Plots S11 and S22 on Smith charts and S21 and S12 in dB magnitude in a combined figure layout.
    
    Usage:
    Initialize the Plot class with either a single Network or a list of Networks, optionally providing names for each.
    Then, call the desired plotting method:
    
    Example:
    plts = SParamPlot(ntw, name='Sample Network')
    plts.plot_smith(0, 0)  # Plots S11 on Smith chart
    plts.plot_db(1, 0)     # Plots S21 in dB scale
    
    For combined plots:
    plts.plot_combined_network_parameters()  # Plots S11, S22 on Smith charts, and S21, S12 in dB
    """
    
    def __init__(self, ntw, name=None):  
        self.ntw = [ntw] if not isinstance(ntw, list) else ntw 
        self.name     = name or [f'Network {i}' for i in range(len(self.ntw_list))]
        
        
    def plot_smith(self,   m, n, title=None):    
       """Plot a single S-parameter on Smith chart."""
       fig, ax = plt.subplots(1, figsize=(10,7))
       for network in self.ntw:
            network.plot_s_smith(m=m, n=n, ax=ax, show_legend=False)
       if title: ax.set_title(title)
       fig.tight_layout()
       plt.show()
    

    def plot_db(self, m, n, title=None):
        """Plot a single S-parameter in dB."""
        fig, ax = plt.subplots(1, figsize=(10,7))
        k=0
        for network in self.ntw:
            frequency = network.f / 1e9
            s_param   = network.s[:, m, n]
            ax.plot(frequency, 20 * np.log10(abs(s_param)), label=self.name[k])
            k+=1
        ax.set_xlabel("Frequency (GHz)", fontsize=18)
        ax.set_ylabel(f"{title} (dB)", fontsize=18)
        ax.set_xlim(np.min(frequency), np.max(frequency))
        ax.legend(loc='best', fontsize=16)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show() 


    def plot_all_s_smith(self):
        """Plot all S-parameters on the same Smith chart."""
        s_param_labels = ['s11', 's22', 's21', 's12']
        line_styles = ['-', '--', '-.', ':']
        
        fig, ax = plt.subplots(1, figsize=(10,7))
        for network, name in zip(self.ntw, self.name):
            for (m, n), ls, label in zip([(0, 0), (1, 1), (0, 1), (1, 0)], line_styles, s_param_labels):
                # Plot each S-parameter with a different line style and label
                network.plot_s_smith(m=m, n=n, ax=ax, ls=ls, label=f'{name} {label}')
            
        fig.tight_layout()
        plt.show()
        

    def plot_all_s_db(self):
        """Plot all S-parameters in dB with labels only on the S12 plot."""
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
        for i, ax in enumerate(axs.flatten()):
            m, n = divmod(i, 2)
            for network, name in zip(self.ntw, self.name):
                frequency = network.f / 1e9
                s_param = network.s[:, m, n]
                if m == 0 and n == 1:  # Check if it's S12
                    ax.plot(frequency, 20 * np.log10(abs(s_param)), label=name)
                else:
                    ax.plot(frequency, 20 * np.log10(abs(s_param)))
                    
            ax.set_xlabel("Frequency (GHz)", fontsize=18)
            ax.set_ylabel(f"S{m+1}{n+1} (dB)", fontsize=18)
            ax.set_xlim(np.min(frequency), np.max(frequency))
            if m == 0 and n == 1:  # Add legend only to S12 plot
                ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
            

    def plot_s_combo(self):
        """
        Plots the S-parameters from a list of networks.
        S11 and S22 are plotted on a Smith chart, while S21 and S12 are plotted in magnitude (dB).
        This is useful for comparing measured and modeled data.
        """
        if not self.ntw:
            raise ValueError("The network list is empty.")
        
        s11_smith, s22_smith, s21_mag, s12_mag, freq = self.extract_s_parameters()

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        self.plot_smith_chart(axes[0, 0], s11_smith, "S11")
        self.plot_smith_chart(axes[0, 1], s22_smith, "S22")
        self.plot_magnitude(axes[1, 0], freq, s21_mag, "|S21| (dB)")
        self.plot_magnitude(axes[1, 1], freq, s12_mag, "|S12| (dB)")

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.show()

    def extract_s_parameters(self):
        s11_smith = [x.s11 for x in self.ntw]
        s22_smith = [x.s22 for x in self.ntw]
        s21_mag = [x.s21.s_db.squeeze() for x in self.ntw]
        s12_mag = [x.s12.s_db.squeeze() for x in self.ntw]
        freq = self.ntw[0].f / 1e9  # Consider making this conversion more flexible
        return s11_smith, s22_smith, s21_mag, s12_mag, freq

    def plot_smith_chart(self, axis, s_parameters, label):
        for s_param in s_parameters:
            s_param.plot_s_smith(ax=axis, show_legend=False)
        axis.text(-0.95, 0.8, label, fontsize=20)  # Consider relative positioning

    def plot_magnitude(self, axis, frequency, magnitudes, ylabel):
        for i, mag in enumerate(magnitudes):
            names = self.name if self.name is not None else np.arange(len(magnitudes))
            axis.plot(frequency, mag, label=f'{names[i]}')
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(np.min(frequency), np.max(frequency))
        axis.legend(loc='best', fontsize=16)
        axis.grid(True, alpha=0.3)       
    
    @property
    def help(self):
        print(self.__doc__)


        

    
    

        
                



    
    



