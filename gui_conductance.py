# -*- coding: utf-8 -*-
"""
Graphical interface for generating and modulating single neuron behavior.
Neuron consists of 4 conductance elements with single activation variables
representing fast -ve, slow +ve, slow -ve, and ultra-slow +ve conductance.

@author: Luka
"""
from gui_utilities import GUI
from neuron_model import Neuron

# Initial conductance parameters
g1 = 0
E_rev1 = 30 # 'sodium'
voff1 = -20
k1 = 0.1

g2 = 0
E_rev2 = -75 # 'potassium'
voff2 = -20
k2 = 0.1

g3 = 0
E_rev3 = 140 # 'calcium'
voff3 = -50
k3 = 0.15

g4 = 0
E_rev4 = -75 # 'potassium'
voff4 = -50
k4 = 0.15

# Define timescales
tf = 0
ts = 30
tus = 20*20

# Define an empty neuron and then interconnect the elements
neuron = Neuron()
R = neuron.add_conductance(1)
i1 = neuron.add_conductance(g1, E_rev1)
x1 = i1.add_gate(k1, voff1, tf)
i2 = neuron.add_conductance(g2, E_rev2)
x2 = i2.add_gate(k2, voff2, ts)
i3 = neuron.add_conductance(g3, E_rev3)
x3 = i3.add_gate(k3, voff3, ts)
i4 = neuron.add_conductance(g4, E_rev4)
x4 = i4.add_gate(k4, voff4, tus)

gui = GUI(neuron, i0 = -50, vmin = -100, vmax = 20, dv = 1.0, time_step = 0.1,
          plot_fixed_point = True, ymin = -120, ymax = 20)

gui.add_IV_curve(neuron, "Fast", tf, [0.1, 0.75, 0.2, 0.2])
gui.add_IV_curve(neuron, "Slow", ts, [0.4, 0.75, 0.2, 0.2])
gui.add_IV_curve(neuron, "Ultraslow", tus, [0.7, 0.75, 0.2, 0.2])

gui.add_label(0.25, 0.34, "Fast -ve")
s1 = gui.add_slider("$g_{max}$", [0.1, 0.3, 0.3, 0.03], 0, 10, g1, i1.update_g_max)
s2 = gui.add_slider("$V_{off}$", [0.1, 0.25, 0.3, 0.03], -75, 0, voff1, x1.update_voff)

gui.add_label(0.25, 0.19, "Slow +ve")
s3 = gui.add_slider("$g_{max}$", [0.1, 0.15, 0.3, 0.03], 0, 10, g2, i2.update_g_max)
s4 = gui.add_slider("$V_{off}$", [0.1, 0.1, 0.3, 0.03], -75, 0, voff2, x2.update_voff)

gui.add_label(0.75, 0.34, "Slow -ve")
s5 = gui.add_slider("$g_{max}$", [0.6, 0.3, 0.3, 0.03], 0, 10, g3, i3.update_g_max)
s6 = gui.add_slider("$V_{off}$", [0.6, 0.25, 0.3, 0.03], -75, 0, voff3, x3.update_voff)

gui.add_label(0.75, 0.19, "UltraSlow +ve")
s7 = gui.add_slider("$g_{max}$", [0.6, 0.15, 0.3, 0.03], 0, 10, g4, i4.update_g_max)
s8 = gui.add_slider("$V_{off}$", [0.6, 0.1, 0.3, 0.03], -75, 0, voff4, x4.update_voff)

s9 = gui.add_iapp_slider([0.1, 0.02, 0.5, 0.03], -100, 0)

b = gui.add_button("Pause", [0.8, 0.02, 0.1, 0.03], gui.pause)

gui.run()