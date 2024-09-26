# Machine Learning-Based Compact Modeling Flow
This repository contains a demo of the machine learning-based compact modeling flow for emerging semiconductor devices proposed in [[1]](#1), [[2]](#2), and [[3]](#3).

## Setup
The modeling flow is based on TensorFlow and QLattice. The former benefits from a CUDA-enabled environment.
To run the demo file `script/NN_fact.ipynb` as a Jupyter notebook, requirements have to be installed.
When using conda, create a conda environment and then install the requirements:

    conda install --file conda_requirements.txt

 Then install the pip requirements in `pip_requirements.txt`:
 

    pip install -r pip_requirements.txt

Please note that the .h5 export feature requires Keras and TensorFlow <= 2.15.0.
 ## Run Flow
   
The demo notebook `script/NN_fact.ipynb` can then be executed e.g. in VS Code. It demonstrates the deep learning based compact modeling of the planar RFET.
After the modeling flow completes, the respective compact model can be found in `script/export/`.

Authors: Maximilian Reuter and Johannes Wilm.

## References

<a id="1">[1]</a> M. Reuter, "Data Driven Compact Modeling of a Reconfigurable FET", Dissertation, 2024, (submitted)

<a id="2">[2]</a> M. Reuter, J. Wilm, A. Kramer, et al., “Machine Learning-Based Compact Model Design for Reconfigurable FETs,” IEEE Journal of the Electron Devices Society, vol. 12, pp. 310–317, 2024. doi: 10.1109/JEDS.2024.3386113.

<a id="3">[3]</a> J. Trommer, M. Reuter, N. Bhattacharjee, Y. He, V. Sessi, M. Drescher, M. Zier, M. Simon, K. Ruttloff, K. Li, A. Zeun, A.-S. Seidel, C. Metze, M. Grothe, S. Jansen, G. Galderisi, V. Havel, S. Slesazeck, J. Hoentschel, K. Hofmann, and T. Mikolajick. "Speeding-Up Emerging Device Development Cycles by Generating Models via Machine-Learning directly from Electrical Measurements". _European Solid-State Electronics Research Conference (ESSERC)_, Bruges, Belgium, (2024), (in press)
