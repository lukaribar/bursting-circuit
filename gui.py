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

from time import time
from collections import deque

from neuron_model import Neuron

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

# Initial constant input current
i_app_const = -2
i_app = lambda t: i_app_const

# Initial values pulse
pulse_on = False
tend = 0

# Initialize pause value
pause_value = False

# Define timescales
tf = 0
ts = 50
tus = 50*50

# Define an empty neuron and then interconnect the elements
neuron = Neuron()
R = neuron.add_conductance(1)
i1 = neuron.add_current(a_f, voff_f, tf) # fast negative conductance
i2 = neuron.add_current(a_s1, voff_s1, ts) # slow positive conductance
i3 = neuron.add_current(a_s2, voff_s2, ts) # slow negative conductance
i4 = neuron.add_current(a_us, voff_us, tus) # ultraslow positive conductance

# **** UPDATE I-V CURVES ****
def update_IV_curves():
    global I_fast, I_slow, I_ultraslow
    
    # Update I-V curves
    I_fast = neuron.IV(V, tf)
    I_slow = neuron.IV(V, ts)
    I_ultraslow = neuron.IV(V, tus)
    
    # Update +/- slope sections
    update_fast_vector()
    update_slow_vector()
    
    # Plot the I-V curves
    plot_fast()
    plot_slow()
    plot_ultraslow()

# **** FUNCTIONS THAT TRACK SLOPE CHANGES IN THE I-V CURVES ******************

def update_fast_vector():
    # Create a list of sections for the fast I-V curve
    
    global fast_vector # List of fast sections
    global fast_index1, fast_index2 # Region of negative conductance
    
    fast_vector = [] 
    
    # Find points where the slope changes
    dIdV = np.diff(I_fast)
    indices = np.where(np.diff(np.sign(dIdV)) != 0)
    indices = indices[0]
    
    indices = np.append(indices, V.size - 1)
    
    # In case the curve is monotone
    fast_index1 = -1
    fast_index2 = -1  

    prev = 0
    slope = dIdV[0] > 0
    
    # Iterate through each section
    for i in np.nditer(indices):
        if slope:
            fast_vector.append([prev, i+2, 'C0']) # Fast +ve
        else:
            fast_vector.append([prev, i+2, 'C3']) # Fast -ve
            fast_index1 = prev
            fast_index2 = i
        slope = not(slope) # Slope changes at each section
        prev = i+1
        

def update_slow_vector():
    # Create a list of sections for the slow/ultra-slow I-V curves
    
    global slow_vector # List of slow sections
    
    slow_vector = [] 
    
    # Find points where slope changes
    dIdV = np.diff(I_slow)
    indices = np.where(np.diff(np.sign(dIdV)) != 0)
    indices = indices[0]
    
    indices = np.append(indices, V.size - 1)
    
    prev = 0
    slope = dIdV[0] > 0
    
    # Iterate through each section    
    for i in np.nditer(indices):
        if slope:
            # Slow +ve, plot regions of fast -ve with different color
            i1 = fast_index1
            i2 = fast_index2
            if i1 < prev:
                i1 = prev-1
            if i1 > i:
                i1 = i-1
            if i2 < prev:
                i2 = prev-1
            if i2 > i:
                i2 = i-1
                
            slow_vector.append([prev, i1+2, 'C0']) # Slow +ve, fast +ve
            slow_vector.append([i1+1, i2+2, 'C3']) # Slow +ve, fast -ve
            slow_vector.append([i2+1, i+2, 'C0']) # Slow +ve, fast +ve            
            
        else:
            slow_vector.append([prev, i+2, 'C1']) # Slow -ve
        slope = not(slope)
        prev = i+1

# **** FUNCTIONS FOR PLOTTING I-V CURVES *************************************
         
def plot_fast():
    axf.cla()
    axf.set_xlabel('V')
    axf.set_ylabel('I')
    axf.set_title('Fast')

    for el in fast_vector:
        i1 = el[0]
        i2 = el[1]
        col = el[2]
        axf.plot(V[i1:i2], I_fast[i1:i2], col)
        
def plot_slow():
    axs.cla()
    axs.set_xlabel('V')
    axs.set_ylabel('I')
    axs.set_title('Slow')

    for el in slow_vector:
        i1 = el[0]
        i2 = el[1]
        col = el[2]
        axs.plot(V[i1:i2], I_slow[i1:i2], col)        

def plot_ultraslow():
    axus.cla()
    axus.set_xlabel('V')
    axus.set_ylabel('I')
    axus.set_title('Ultra-slow')

    for el in slow_vector:
        i1 = el[0]
        i2 = el[1]
        col = el[2]
        axus.plot(V[i1:i2], I_ultraslow[i1:i2], col)
    
    axus.plot(V, np.ones(len(V)) * i_app_const,'C2')

# **** FUNCTIONS TO UPDATE PARAMETERS ON GUI CHANGES *************************

def update_iapp(val):
    global i_app_const, i_app
    i_app_const = val
    i_app = lambda t: i_app_const
    
    update_IV_curves()

def update_val(val, update_method):
    update_method(val)
    update_IV_curves()
    
def pulse(event):
    global pulse_on, tend, i_app
    
    # Pulse parameters
    delta_t = 10
    delta_i = 1
    
    tend = t + delta_t
    pulse_on = True
    
    i_app = lambda t: (i_app_const + delta_i)
    
def pause(event):
    global pause_value
    pause_value = not(pause_value)
    if pause_value:
        button_pause.label.set_text('Resume')
    else:
        button_pause.label.set_text('Pause')

# **** DRAW GRAPHICAL USER INTERFACE *****************************************

# Close pre-existing figures
plt.close("all")

fig = plt.figure()

# Fast I-V curve
axf = fig.add_subplot(2, 3, 1)
axf.set_position([0.1, 0.75, 0.2, 0.2])

# Slow I-V curve
axs = fig.add_subplot(2, 3, 2)
axs.set_position([0.4, 0.75, 0.2, 0.2])

# Ultraslow I-V curve
axus = fig.add_subplot(2, 3, 3)
axus.set_position([0.7, 0.75, 0.2, 0.2])

# Time - Voltage plot
axsim = fig.add_subplot(2, 3, 4)
axsim.set_position([0.1, 0.45, 0.8, 0.2])
axsim.set_ylim((-5, 5))
axsim.set_xlabel('Time')
axsim.set_ylabel('V')

# Sliders for fast negative conductance
axf1 = plt.axes([0.1, 0.3, 0.3, 0.03])
slider_i1_a = Slider(axf1, 'Gain', 0, 4, valinit = -a_f)
slider_i1_a.on_changed(lambda val: update_val(-val, i1.update_a))

axf2 = plt.axes([0.1, 0.25, 0.3, 0.03])
slider_i1_voff = Slider(axf2, '$V_{off}$', -2, 2, valinit = voff_f)
slider_i1_voff.on_changed(lambda val: update_val(val, i1.update_voff))

# Sliders for slow positive conductance
axs11 = plt.axes([0.1, 0.15, 0.3, 0.03])
slider_i2_a = Slider(axs11, 'Gain', 0, 4, valinit = a_s1)
slider_i2_a.on_changed(lambda val: update_val(val, i2.update_a))

axs12 = plt.axes([0.1, 0.1, 0.3, 0.03])
slider_i2_voff = Slider(axs12, '$V_{off}$', -2, 2, valinit = voff_s1)
slider_i2_voff.on_changed(lambda val: update_val(val, i2.update_voff))

# Sliders for slow negative conductance
axs21 = plt.axes([0.6, 0.3, 0.3, 0.03])
slider_i3_a = Slider(axs21, 'Gain', 0, 4, valinit = -a_s2)
slider_i3_a.on_changed(lambda val: update_val(-val, i3.update_a))

axs22 = plt.axes([0.6, 0.25, 0.3, 0.03])
slider_i3_voff = Slider(axs22, '$V_{off}$', -2, 2, valinit = voff_s2)
slider_i3_voff.on_changed(lambda val: update_val(val, i3.update_voff))

# Sliders for ultraslow positive conductance
axus1 = plt.axes([0.6, 0.15, 0.3, 0.03])
slider_i4_a = Slider(axus1, 'Gain', 0, 4, valinit = a_us)
slider_i4_a.on_changed(lambda val: update_val(val, i4.update_a))

axus2 = plt.axes([0.6, 0.1, 0.3, 0.03])
slider_i4_voff = Slider(axus2, '$V_{off}$', -2, 2, valinit = voff_us)
slider_i4_voff.on_changed(lambda val: update_val(val, i4.update_voff))

# Slider for Iapp
axiapp = plt.axes([0.1, 0.02, 0.5, 0.03])
slider_iapp = Slider(axiapp, '$I_{app}$',-3, 3, valinit = i_app_const)
slider_iapp.on_changed(update_iapp)

# Button for I_app = pulse(t)
axpulse_button = plt.axes([.675, 0.02, 0.1, 0.03])
pulse_button = Button(axpulse_button, 'Pulse')
pulse_button.on_clicked(pulse)

# Button for pausing the simulation
axbutton = plt.axes([0.8, 0.02, 0.1, 0.03])
button_pause = Button(axbutton, 'Pause')
button_pause.on_clicked(pause)

# Labels for conductance sliders
plt.figtext(0.25, 0.34, 'Fast -ve', horizontalalignment = 'center')
plt.figtext(0.25, 0.19, 'Slow +ve', horizontalalignment = 'center')
plt.figtext(0.75, 0.34, 'Slow -ve', horizontalalignment = 'center')
plt.figtext(0.75, 0.19, 'Ultraslow +ve', horizontalalignment = 'center')

# **** PLOT THE I-V CURVES AND SIMULATION DATA *******************************

# Plot I-V curves
V = np.arange(-3,3.1,0.1)
update_IV_curves()

# Live simulation
t0 = 0
v0 = neuron.get_init_conditions()

sstep = 100 # draw sstep length of data in a single call
tint = 5000 # time window plotted

tdata, ydata = deque(), deque()
simuln, = axsim.plot(tdata, ydata)

# Define ODE equation for the solvers
def odesys(t, y):
    return neuron.sys(i_app(t),y)

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
while plt.fignum_exists(fig.number):
    while pause_value:
        plt.pause(0.01)
    
    st = time()

    last_t = t
    
    # Check for pulse
    while t - last_t < sstep:
        if pulse_on and (t > tend):
            i_app = lambda t: i_app_const
            pulse_on = False
        
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
    axsim.set_xlim(tdata[-1] - tint, tdata[-1] + tint / 20)
    fig.canvas.draw()
    fig.canvas.flush_events()