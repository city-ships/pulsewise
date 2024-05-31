Here's an updated `README.md` file that includes the full software information required to run the script:

```markdown
# Laser-Induced Phase Transformations in Sb2S3 Thin Films

## Overview
This repository contains a Python script for simulating the heating and phase change dynamics in Sb2S3 thin films induced by pulsed laser irradiation. The simulation framework integrates critical physical phenomena such as thin film interference, heat conduction, and temperature-dependent optical properties to provide a comprehensive analysis of the laser-induced phase change process.

## Features
- **Thin Film Interference**: Accounts for the interaction of laser light with the thin film stack, affecting the distribution of the electric field and absorption within the film.
- **Heat Conduction**: Models the spread of heat through the material, primarily in the direction normal to the film surface.
- **Temperature-Dependent Properties**: Incorporates changes in optical and thermal properties with temperature.
- **Melting and Solidification**: Simulates the transition from solid to liquid upon heating and back to solid upon cooling, including the specific heat of melting.

## Installation
To run the simulation, you need to have Python 3.10.12 installed along with the following dependencies:

### Python Packages
- numpy 1.21.6
- numba 0.55.1
- matplotlib 3.7.2
- scipy 1.11.2
- fenics-dolfinx 0.6.0
- fenics-ufl 2023.1.1.post0
- mpi4py 3.1.3
- petsc4py 3.15.1
- joblib 1.3.2
- bayesian-optimization 1.4.3

You can install these dependencies using `pip`:
```bash
pip install numpy==1.21.6 numba==0.55.1 matplotlib==3.7.2 scipy==1.11.2 fenics-dolfinx==0.6.0 fenics-ufl==2023.1.1.post0 mpi4py==3.1.3 petsc4py==3.15.1 joblib==1.3.2 bayesian-optimization==1.4.3
```

### OS Information
- **Linux Kernel**: 5.15.0-79-generic x86_64
- **Distro**: Linux Mint 21.2 Victoria

## Usage
The main script is `laser_phase_transformation_simulation.py`. To run the simulation, use the following command:

```bash
python laser_phase_transformation_simulation.py
```

### Example
Here is an example of how to run the simulation with specific parameters:

```python
if __name__ == "__main__":
    run_simulation(end=1.389e-9, a1=1, a2=1)  # spot 16
```

## Script Details
The script simulates the heating and phase change dynamics in Sb2S3 thin films under pulsed laser irradiation. It calculates the absorbed power using interference patterns, updates the temperature distribution based on the absorbed power and heat conduction, and recalculates the optical and thermal properties for each time step.

## Applications
- **Optimization**: The framework can be used to optimize laser parameters for efficient phase change, minimizing energy consumption, and maximizing the speed of amorphization.
- **Adaptability**: It can be adapted for various thin film materials and laser configurations, making it a versatile tool for studying laser-induced phase transformations.

## Citation
If you use this code in your research, please cite the following paper:

Resl, J., Hingerl, K., Gutierrez, Y., Losurdo, M., & Cobet, C. (2024). Optimizing Laser-Induced Phase Transformations in Sb2S3 Thin Films: Simulation Framework and Experiments. *Journal Name*. DOI: XXXXXXX

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
The authors acknowledge the support from the European Union’s Horizon 2020 research and innovation program (No 899598 - PHEMTRONICS), the Ramon y Cajal Fellowship (RYC2022-037828-I), and the Danube Project (Project No. MULT 07/2023).
```

This `README.md` now includes specific details about the Python packages and their versions, as well as the OS information required to run the script.
