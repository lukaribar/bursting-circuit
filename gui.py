# -*- coding: utf-8 -*-
"""
Graphical interface for controlling single neuron behavior.
Neuron consists of 4 current source elements representing fast -ve, slow +ve,
slow -ve, and ultra-slow +ve conductance.

@author: Luka
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

from collections import deque

from neuron_model import Neuron

class IV_curve:
    """
    Describe the class
    """
    class Segment():
        def __init__(self, start, end, color):
            self.start = start
            self.end = end
            self.color = color
    
    def __init__(self, neuron, timescale, V, cols):
        self.neuron = neuron
        self.timescale = timescale
        self.V = V
        self.cols = cols
        self.segments = []
                    
        # Calculate the IV curve and the corresponding segments
        self.update()
    
    def update(self, prev_segments = []):
        # If no preceeding IV curves, put [Vmin, Vmax] as prev_segment
        if (prev_segments == []):
            prev_segments = [self.Segment(0, self.V.size-1, self.cols[0])]
        
        self.I = self.neuron.IV(self.V, self.timescale)
        
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


# DEFINE A CLASS WITH ALL PLOTTING FUNCTIONALITY
class GUI:
    """
    Describe the class
    """
    _params = {'vmin': -3, 'vmax': 3.1, 'dv': 0.1}
                 
    def __init__(self, neuron, **kwargs):
        self.__dict__.update(self._params) # Default parameters
        self.__dict__.update(kwargs) # Modify parameters
        
        # Colors of the +ve/-ve conductance regions
        # First is +ve conductance, each successive is -ve conductance in next
        # timescale
        self.colors = ['C0', 'C3', 'C1', 'C6']
        
        self.neuron = neuron # associate GUI with a neuron
        
        self.V = np.arange(self.vmin,self.vmax,self.dv)
        self.i_app_const = 0
        self.i_app = lambda t: self.i_app_const
        
        self.IV_curves = []
        self.IV_size = 0
        
        # Create empty plot
        plt.close("all")
        self.fig = plt.figure()
        self.axs_iv = []
        
        # Add simulation plot
        self.axsim = self.fig.add_subplot(2, 3, 4)
        self.axsim.set_position([0.1, 0.45, 0.8, 0.2]) # move this 
        self.axsim.set_ylim((-5, 5))
        self.axsim.set_xlabel('Time')
        self.axsim.set_ylabel('V')
        
    def add_IV_curve(self, name, timescale, coords):
        self.IV_size += 1
        ax = self.fig.add_subplot(2, 3, self.IV_size)
        ax.set_position(coords)
        ax.set_xlabel('V')
        ax.set_ylabel('I')
        ax.set_title(name)
        
        self.axs_iv.append(ax)
        
        self.IV_curves.append(IV_curve(self.neuron, timescale, self.V,
                                            [self.colors[0],
                                             self.colors[self.IV_size]]))
        
        self.update_IV_curves()
        
    def update_IV_curves(self):
        for idx, (iv_curve, ax) in enumerate(zip(self.IV_curves, self.axs_iv)):
            # Update the segments
            if (idx > 0):
                prev_segments = self.IV_curves[idx-1].get_segments()
                iv_curve.update(prev_segments)
            else:
                iv_curve.update()
            
            # Plot
            ax.cla()
            for s in iv_curve.segments:
                i1 = s.start
                i2 = s.end
                col = s.color
                # Add +1 to end points to include them in the plot
                ax.plot(self.V[i1:i2+1], iv_curve.I[i1:i2+1], col)
                
        # Add Iapp line to the last IV curve
        self.axs_iv[-1].plot(self.V, np.ones(len(self.V)) * self.i_app_const,
                   'C2')
    
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
        
    def add_button(self, name, coords, on_press_method):
        ax = plt.axes(coords)
        button = Button(ax, name)
        button.on_clicked(on_press_method)
        return button
        
    def add_label(self, x, y, text):
        plt.figtext(x, y, text, horizontalalignment = 'center')
        
    def run(self):
        t0 = 0
        v0 = self.neuron.get_init_conditions()
        
        sstep = 100 # draw sstep length of data in a single call
        tint = 5000 # time window plotted
        
        tdata, ydata = deque(), deque()
        simuln, = self.axsim.plot(tdata, ydata)
        
        # Define ODE equation for the solvers
        def odesys(t, y):
            return neuron.sys(self.i_app(t),y)
        
        # Standard ODE solvers (RK45, BDF, etc) (import from scipy.integrate)
        #solver = BDF(odesys, 0, v0, np.inf, max_step=sstep)
        #y = solver.y
        #t = solver.t
        
        # Basic Euler step
        y = v0
        t = t0
        
        def euler_step(odesys, t0, y0):
            dt = 1 # step size
            y = y0 + odesys(t0,y0)*dt
            t = t0 + dt
            return t, y
         
        # Comment Euler step or standard solver step depending on the method
        while plt.fignum_exists(self.fig.number):
            #while pause_value:
            #    plt.pause(0.01)
        
            last_t = t
            
            while t - last_t < sstep:
                #if pulse_on and (t > tend):
                #    i_app = lambda t: i_app_const
                #    pulse_on = False
                
                # Euler step
                t,y = euler_step(odesys,t,y)
                
                # Standard solver step
        #        msg = solver.step()
        #        t = solver.t
        #        y = solver.y
        #        if msg:
        #            raise ValueError('solver terminated with message: %s ' % msg)
                
                tdata.append(t)
                ydata.append(y[0])
        
            while tdata[-1] - tdata[0] > 2 * tint:
                tdata.popleft()
                ydata.popleft()
        
            simuln.set_data(tdata, ydata)
            self.axsim.set_xlim(tdata[-1] - tint, tdata[-1] + tint / 20)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    class IV_curve:
        """
        Describe the class
        """
        class Segment():
            def __init__(self, start, end, color):
                self.start = start
                self.end = end
                self.color = color
        
        def __init__(self, neuron, timescale, V, cols):
            self.neuron = neuron
            self.timescale = timescale
            self.V = V
            self.cols = cols
            self.segments = []
                        
            # Calculate the IV curve and the corresponding segments
            self.update()
        
        def update(self, prev_segments = []):
            # If no preceeding IV curves, put [Vmin, Vmax] as prev_segment
            if (prev_segments == []):
                prev_segments = [self.Segment(0, self.V.size-1, self.cols[0])]
            
            self.I = self.neuron.IV(self.V, self.timescale)
            
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
                    if (prev_segment.start <= new_segment.end
                        <= prev_segment.end):
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

plt.ion() # turn the interactive mode on

# **** DEFINE INITIAL PARAMETERS AND NEURON MODEL ****************************

# Initial conductance parameters
a_f = -2
voff_f = 0
a_s1 = 2
voff_s1 = 0
a_s2 = -1.5
voff_s2 = -0.9
a_us = 1.5
voff_us = -0.9

# Define timescales
tf = 0
ts = 50
tus = 50*50

# Move this?
i_app_const = 0

# Define an empty neuron and then interconnect the elements
neuron = Neuron()
R = neuron.add_conductance(1)
i1 = neuron.add_current(a_f, voff_f, tf) # fast negative conductance
i2 = neuron.add_current(a_s1, voff_s1, ts) # slow positive conductance
i3 = neuron.add_current(a_s2, voff_s2, ts) # slow negative conductance
i4 = neuron.add_current(a_us, voff_us, tus) # ultraslow positive conductance

gui = GUI(neuron)

gui.add_IV_curve("Fast", tf, [0.1, 0.75, 0.2, 0.2])
gui.add_IV_curve("Slow", ts, [0.4, 0.75, 0.2, 0.2])
gui.add_IV_curve("Ultraslow", tus, [0.7, 0.75, 0.2, 0.2])

gui.add_label(0.25, 0.34, "Fast -ve")
s1 = gui.add_slider("Gain", [0.1, 0.3, 0.3, 0.03], 0, 4, a_f, i1.update_a, sign=-1)
s2 = gui.add_slider("$V_{off}$", [0.1, 0.25, 0.3, 0.03], -2, 2, voff_f, i1.update_voff)

gui.add_label(0.25, 0.19, "Slow +ve")
s3 = gui.add_slider("Gain", [0.1, 0.15, 0.3, 0.03], 0, 4, a_s1, i2.update_a)
s4 = gui.add_slider("$V_{off}$", [0.1, 0.1, 0.3, 0.03], -2, 2, voff_s1, i2.update_voff)

gui.add_label(0.75, 0.34, "Slow -ve")
s5 = gui.add_slider("Gain", [0.6, 0.3, 0.3, 0.03], 0, 4, a_s2, i3.update_a, sign=-1)
s6 = gui.add_slider("$V_{off}$", [0.6, 0.25, 0.3, 0.03], -2, 2, voff_s2, i3.update_voff)

gui.add_label(0.75, 0.19, "UltraSlow +ve")
s7 = gui.add_slider("Gain", [0.6, 0.15, 0.3, 0.03], 0, 4, a_us, i4.update_a)
s8 = gui.add_slider("$V_{off}$", [0.6, 0.1, 0.3, 0.03], -2, 2, voff_us, i4.update_voff)

s9 = gui.add_slider("$I_{app}$", [0.1, 0.02, 0.5, 0.03], -3, 3, i_app_const, gui.update_iapp)

gui.run()


# **** FUNCTIONS TO UPDATE PARAMETERS ON GUI CHANGES *************************

# ADD THIS
    
#def pulse(event):
#    global pulse_on, tend, i_app
#    
#    # Pulse parameters
#    delta_t = 10
#    delta_i = 1
#    
#    tend = t + delta_t
#    pulse_on = True
#    
#    i_app = lambda t: (i_app_const + delta_i)
#    
#def pause(event):
#    global pause_value
#    pause_value = not(pause_value)
#    if pause_value:
#        button_pause.label.set_text('Resume')
#    else:
#        button_pause.label.set_text('Pause')

# **** DRAW GRAPHICAL USER INTERFACE *****************************************

# Button for I_app = pulse(t)
#axpulse_button = plt.axes([.675, 0.02, 0.1, 0.03])
#pulse_button = Button(axpulse_button, 'Pulse')
#pulse_button.on_clicked(pulse)

# Button for pausing the simulation
#axbutton = plt.axes([0.8, 0.02, 0.1, 0.03])
#button_pause = Button(axbutton, 'Pause')
#button_pause.on_clicked(pause)


