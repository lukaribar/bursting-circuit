# -*- coding: utf-8 -*-
"""
Classes and methods for defining a graphical user interface for neural
simulations

@author: Luka
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

from collections import deque

class IV_curve:
    """
    IV curve class with added functionality of finding regions of negative
    conductance and associating the corresponding color scheme
    
    args:
        neuron: associated neuron whose IV curve is calculated
        name: name of the IV curve, used in plotting
        timescale: timescale of the IV curve
        V: voltage range used for the IV curve
        cols: coloring scheme -> cols[0] = positive conductance
                              -> cols[1] = negative conductance
    methods:
        update: calculate segments used for plotting, by inserting negative
        conductance segments into the segments of the preceeding IV curve
        (prev_segments)
        If IV curve corresponds to the fastest timescale no prev_segments are
        passed
    """
    class Segment():
        """
        Individual segments to be plotted
        
        args:
            start, end: Vstart and Vend indices for the segment
            color: color of the segment in the plot
        """
        def __init__(self, start, end, color):
            self.start = start
            self.end = end
            self.color = color
    
    def __init__(self, neuron, name, timescale, V, cols):
        self.neuron = neuron
        self.name = name
        self.timescale = timescale
        self.V = V
        self.I = []
        self.cols = cols
        self.segments = []
                        
    def update(self, vrest, prev_segments = []):
        # If no preceeding IV curves, put [Vmin, Vmax] as prev_segment
        if (prev_segments == []):
            prev_segments = [self.Segment(0, self.V.size-1, self.cols[0])]
        
        self.I = self.neuron.IV(self.V, self.timescale, vrest)
        
        col = self.cols[1] # color for -ve conductance
        
        # Find regions of -ve conductance
        dIdV = np.diff(self.I)
        indices = np.where(np.diff(np.sign(dIdV)) != 0)
        indices = indices[0] + 1 # +1 for the correct max/min points
        indices = np.append(indices, self.V.size) # add ending point
        
        slope = dIdV[0] < 0 # True if initial slope -ve
        
        prev = 0
        
        new_segments = []
        
        # Get regions of -ve conductance
        for i in np.nditer(indices):
            # If region of -ve conductance
            if slope:
                new_segments.append(self.Segment(prev, i, col))
            slope = not(slope) # Sign changes after every point in indices
            prev = i
            
        # Insert new segments
        for new_segment in new_segments:
            # Find which prev_segments containt new_segment start and end
            for idx, prev_segment in enumerate(prev_segments):
                if (prev_segment.start <= new_segment.start
                    <= prev_segment.end):
                    idx1 = idx
                    col1 = prev_segment.color
                    start1 = prev_segment.start
                if (prev_segment.start <= new_segment.end <= prev_segment.end):
                    idx2 = idx
                    col3 = prev_segment.color
                    end3 = prev_segment.end
            
            # Delete the old segments between idx1 and idx2
            del prev_segments[idx1:idx2+1]
            
            # start and end variables of new segments to insert
            end1 = new_segment.start
            start3 = new_segment.end
            
            # Insert new segments
            prev_segments.insert(idx1, self.Segment(start3, end3, col3))
            prev_segments.insert(idx1, new_segment)
            prev_segments.insert(idx1, self.Segment(start1, end1, col1))
                
        self.segments = prev_segments
        
    def get_segments(self):
        return self.segments
    
    def get_I(self):
        return self.I

# DEFINE A CLASS WITH ALL PLOTTING FUNCTIONALITY
class GUI:
    """
    Graphical user interface class with methods for plotting the IV curves and
    simulation results along with methods for adding sliders and buttons for
    changing the neuronal parameters.
    
    args:
        system: Neuron or Network class to be simulated
    
    kwargs:
        vmin, vmax, dv: voltage range of the IV curves
        i0: initial applied current
        plot_fixed_point: set to True to plot the fixed point in the I_ss
        time_step: step size for the Euler solver
        ymin, ymax: voltage range for the time plot
        sstep: wait for sstep data points before updating the time plot
        tint: length of the time plot
    """
    _params = {'vmin': -3, 'vmax': 3.1, 'dv': 0.1, 'i0': 0,
               'plot_fixed_point': False, 'time_step': 1,
               'ymin': -5, 'ymax': 5, 'sstep': 100, 'tint': 5000}
                 
    def __init__(self, system, **kwargs):
        self.__dict__.update(self._params) # Default parameters
        self.__dict__.update(kwargs) # Modify parameters
        
        # Colors of the +ve/-ve conductance regions
        # First is +ve conductance, each successive is -ve conductance in next
        # timescale
        self.colors = ['C0', 'C3', 'C1', 'C6']
        
        self.system = system # associate GUI with a neuron or a network
        
        self.V = np.arange(self.vmin,self.vmax,self.dv)
        
        # Extended voltage range for finding vrest
        vrange = self.vmax - self.vmin
        self.V_extended = np.arange(self.vmin-vrange/2, self.vmax+vrange/2, 
                                    self.dv)
        
        self.i_app_const = self.i0
        self.i_app = lambda t: self.i_app_const
        
        self.IV_curves = []
        self.IV_size = 0
        
        self.v_rest = 0
        self.I_ss_rest = 0
        
        # Create empty plot
        plt.close("all")
        self.fig = plt.figure()
        self.axs_iv = []
        
        # Add simulation plot
        self.axsim = self.fig.add_subplot(2, 3, 4)
        self.axsim.set_position([0.1, 0.45, 0.8, 0.2]) # move this 
        self.axsim.set_ylim((self.ymin, self.ymax))
        self.axsim.set_xlabel('Time')
        self.axsim.set_ylabel('V')
        
        self.pause_value = False
        
    def add_IV_curve(self, neuron, name, timescale, coords):
        self.IV_size += 1
        ax = self.fig.add_subplot(2, 3, self.IV_size)
        ax.set_position(coords)
        ax.set_xlabel('V')
        ax.set_ylabel('I')
        ax.set_title(name)
        
        self.axs_iv.append(ax)
        
        self.IV_curves.append(IV_curve(neuron, name, timescale, self.V,
                                            [self.colors[0],
                                             self.colors[self.IV_size]]))
        
        self.update_IV_curves()
        
    def update_IV_curves(self):
        # Update v_rest
        self.update_fixed_point()
        
        for idx, (iv_curve, ax) in enumerate(zip(self.IV_curves, self.axs_iv)):
            # Update the segments
            if (idx > 0):
                prev_segments = self.IV_curves[idx-1].get_segments()
            else:
                prev_segments = []
            
            iv_curve.update(self.v_rest, prev_segments)    
            
            # Plot
            ax.cla()
            ax.set_xlabel('V')
            ax.set_ylabel('I')
            ax.set_title(iv_curve.name)
            for s in iv_curve.segments:
                i1 = s.start
                i2 = s.end
                col = s.color
                # Add +1 to end points to include them in the plot
                ax.plot(self.V[i1:i2+1], iv_curve.I[i1:i2+1], col)
                
        # Add Iapp line to the last IV curve
        self.axs_iv[-1].plot(self.V, np.ones(len(self.V)) * self.i_app_const,
                   'C2')
        # Add fixed point circle
        if (self.plot_fixed_point and (self.vmin < self.v_rest < self.vmax)):
            self.axs_iv[-1].plot(self.v_rest,self.I_ss_rest,'C2', marker = '.',
                   markersize = 10)
        
    def update_fixed_point(self):
        I_ss = self.system.IV_ss(self.V_extended)
        zero_crossings = np.where(np.diff(np.sign(I_ss-self.i_app_const)))[0]
        if (zero_crossings.size != 0):
            # Take the equilibrium nearest to the previous equilibrium
            vdiff = abs(self.V_extended[zero_crossings] - self.v_rest)
            index = zero_crossings[vdiff.argmin()]
            #index = zero_crossings[0] # the most left one
            self.v_rest = (self.V_extended[index] + self.V_extended[index+1])/2
            self.I_ss_rest = (I_ss[index] + I_ss[index+1])/2
    
    def update_iapp(self, val):
        self.i_app_const = val
        self.i_app = lambda t: val
        self.update_IV_curves()
        
    def update_val(self, val, update_method):
        update_method(val)
        self.update_IV_curves()
        
    def add_slider(self, name, coords, val_min, val_max, val_init,
                   update_method, sign = 1):
        ax = plt.axes(coords)
        slider = Slider(ax, name, val_min, val_max, valinit = sign*val_init)
        slider.on_changed(lambda val: self.update_val(sign*val, update_method))
        return slider
    
    def add_iapp_slider(self, coords, val_min, val_max):
        slider = self.add_slider("$I_{app}$", coords, val_min, val_max, self.i0, 
                                 self.update_iapp)
        return slider
        
    def add_button(self, name, coords, on_press_method):
        ax = plt.axes(coords)
        button = Button(ax, name)
        button.on_clicked(on_press_method)
        return button
        
    def add_label(self, x, y, text):
        plt.figtext(x, y, text, horizontalalignment = 'center')
     
    def pause(self, event):
        self.pause_value = not(self.pause_value)
    
    def run(self, idx_list = [0]):        
        sstep = self.sstep
        tint = self.tint
                
        tdata = deque()
        ydata_list = []
        simuln_list = []
        for idx in idx_list:
            ydata = deque()
            line, = self.axsim.plot(tdata, ydata)
            simuln_list.append(line)
            ydata_list.append(ydata)
        
        # Set the simulation solver
        t = 0
        self.system.set_solver("Euler", self.i_app, t, sstep, dt = self.time_step)
        
        while plt.fignum_exists(self.fig.number):
            while self.pause_value:
                plt.pause(0.01)
        
            last_t = t
            
            while t - last_t < sstep:                
                # Simulation step
                t, y = self.system.step()
                
                tdata.append(t)
                for i, idx in enumerate(idx_list):
                    ydata_list[i].append(y[idx])
        
            while tdata[-1] - tdata[0] > 2 * tint:
                tdata.popleft()
                for ydata in ydata_list:
                    ydata.popleft()
        
            for i, idx in enumerate(idx_list):
                simuln_list[i].set_data(tdata, ydata_list[i])
            self.axsim.set_xlim(tdata[-1] - tint, tdata[-1] + tint / 20)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()