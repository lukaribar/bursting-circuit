"""
An example of a single-neuron simulation
"""

import matplotlib.pyplot as plt

from neuron_model import Neuron

# Define timescales
tf = 0
ts = 50
tus = 50 * 50

# Define an empty neuron and then interconnect the elements
neuron = Neuron()
R = neuron.add_conductance(1)
i1 = neuron.add_current(-2, 0, tf)  # fast negative conductance
i2 = neuron.add_current(2, 0, ts)  # slow positive conductance
i3 = neuron.add_current(-1.5, -1.5, ts)  # slow negative conductance
i4 = neuron.add_current(1.5, -1.5, tus)  # ultraslow positive conductance

# Simulate the circuit
sol = neuron.simulate(trange=(0, 10000), i_app=lambda t: -2)

# Plot simulation
# y[0] - membrane voltage, y[1] - slow voltage, y[2] - ultra-slow voltage
plt.figure()
plt.plot(sol.t, sol.y[0], sol.t, sol.y[1], sol.t, sol.y[2])
plt.show()
