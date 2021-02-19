# Neuromorphic Neuromodulation
Python implementation of the neuromorphic neuronal model described in:

- [Neuromodulation of Neuromorphic Circuits](https://arxiv.org/abs/1805.05696) (Ribar and Sepulchre, IEEE Transactions on Circuits and Systems, 2019)
- [Neuromorphic Control](https://arxiv.org/abs/2011.04441) (Ribar and Sepulchre, 2020)

## Overview
### Model definition
- `neuron_model.py`
- `network_model.py`

The file `neuron_model.py` provides the definitions for both the current-based model described in [Ribar and Sepulchre, 2019](https://arxiv.org/abs/1805.05696) and the conductance-based model outlined in [Ribar and Sepulchre, 2020](https://arxiv.org/abs/2011.04441).

A neuron is defined as an interconnection of an arbitrary number of either current source or conductance elements. Each conductance element is defined with a collection of gating variables defining the activation and inactivation dynamics. The dynamics of the current elements, as well as the gating variables, are given by linear first-order filters defined by their timescale.

The file `network_model.py` provides the corresponding definitions for the current-based and conductance-based synaptic connections, along with resistive connections.

A neural network is defined as an arbitrary collection of neurons as defined in `neuron_model.py` and a collection of synapses/resistive connections with their corresponding connectivity matrices.

### Graphical interface
- `gui.py`

The I-V curve shaping graphical interface for controlling the neuronal behavior as detailed in [Ribar and Sepulchre, 2019](https://arxiv.org/abs/1805.05696). The file provides an interface for controlling the parameters of the 4-current bursting model with a live plot of the behavior and the corresponding I-V curves.

Additionally, a graphical interface for controlling an equivalent conductance-based model ([Ribar and Sepulchre, 2020](https://arxiv.org/abs/2011.04441)) with 4 activating conductances is provided in `gui_conductance.py`.

### Examples
- `single_neuron_example`
- `network_example`

The examples show how the model definitions are used to construct neurons and networks of neurons.

### Requirements
- `requirements.txt`

## Reference
```
@ARTICLE{ribar2019neuromodulation,
author={L. {Ribar} and R. {Sepulchre}},
journal={IEEE Transactions on Circuits and Systems I: Regular Papers},
title={Neuromodulation of Neuromorphic Circuits},
year={2019},
volume={66},
number={8},
pages={3028-3040},
doi={10.1109/TCSI.2019.2907113},
ISSN={1549-8328},
month={Aug},}
```
```
@ARTICLE{ribar2020neuromorphic,
title={Neuromorphic Control}, 
author={L. {Ribar} and R. {Sepulchre}},
year={2020},
eprint={2011.04441},
archivePrefix={arXiv},
primaryClass={eess.SY}}
```
## License
MIT
