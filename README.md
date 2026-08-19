# Fox-Li LASER Cavity Simulator

A Python-based numerical simulator for studying transverse laser-cavity modes using the **Fox-Li iterative method**.

The project models realistic two-mirror resonators with finite apertures, curved/conic mirror surfaces, mirror misalignment, diffraction, and spatially non-uniform gain. It provides both a NumPy-based implementation and a JAX-based implementation for accelerated numerical computation.

## Problem Being Solved

The transverse mode of a laser cavity is determined by repeated round trips of the optical field through the resonator. For ideal cavities, analytical Gaussian-beam theory can describe the fundamental mode. However, finite mirror apertures, mirror misalignment, non-spherical surfaces, diffraction, and non-uniform gain can produce strongly distorted and non-Gaussian modes that are difficult to obtain analytically.

This project numerically solves for the dominant self-consistent cavity field by repeatedly propagating an initial complex optical field through the resonator until the Fox-Li iteration approaches a stable mode.

The resulting field can then be analyzed in both the **near field and far field**, including beam-quality estimation through \(M^2\).

## Key Features

- Fox-Li iterative cavity-mode simulation
- FFT-based angular-spectrum diffraction propagation
- Finite circular mirror apertures
- Spherical/conic mirror surfaces
- Mirror decenter and angular misalignment
- Spatially varying gain profiles from external data
- Near-field intensity and phase visualization
- Far-field diffraction analysis
- Beam-quality/\(M^2\) estimation
- PyQt5 graphical interface
- NumPy CPU implementation
- JAX-based implementation for computational acceleration

## Versions

### Version 1 — NumPy

The reference implementation using NumPy/SciPy for the numerical calculations.

### Version 2 — JAX

A JAX-based implementation designed to improve computational performance and provide a path toward CPU/GPU acceleration.

## Typical Workflow

```text
Cavity Parameters
       ↓
Mirror & Gain Definition
       ↓
Initial Optical Field
       ↓
Fox-Li Round-Trip Iteration
       ↓
Converged Transverse Mode
       ↓
Near-Field Analysis
       ↓
Far-Field Analysis
       ↓
Beam Quality / M²
```

## Installation

Clone the repository:

```bash
git clone https://github.com/debanjanh5402/Fox-Li-LASER-Cavity-Simulator.git
cd Fox-Li-LASER-Cavity-Simulator
```

For Version 1:

```bash
cd v1
python -m pip install -r requirements_macos.txt
cd source
python main.py
```

For Version 2:

```bash
cd v2
python -m pip install -r requirements_macos.txt
cd source
python main.py
```

Use the corresponding requirements file for other supported platforms.

## Repository Structure

```text
Fox-Li-LASER-Cavity-Simulator/
├── v1/                    # NumPy implementation
├── v2/                    # JAX implementation
├── results/               # Simulation results
├── BUILD.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── README.md       
```

## Applications

The simulator can be used to investigate:

- cavity transverse-mode formation;
- diffraction and aperture losses;
- mirror misalignment;
- non-Gaussian cavity modes;
- gain-induced mode shaping;
- far-field beam structure;
- beam-quality degradation;
- effects of realistic cavity imperfections.

## Limitations

The current implementation is primarily a scalar numerical optics model. Polarization, full vector electromagnetic effects, detailed gain saturation, thermal dynamics, and complete longitudinal laser dynamics are not yet modeled self-consistently.

## Author

**Debanjan Halder**  
Research Scientist  
Computational Optics Laboratory  
Optics & Photonics Centre (OPC)  
Indian Institute of Technology Delhi

Developed under the supervision of **Prof. Kedar B. Khare**, IIT Delhi.

## Citation

If you use this simulator in academic work, please cite:

```text
Debanjan Halder.
Fox-Li LASER Cavity Simulator.
2026.
```

See `CITATION.cff` for machine-readable citation metadata.

## License

MIT License. See `LICENSE`.
