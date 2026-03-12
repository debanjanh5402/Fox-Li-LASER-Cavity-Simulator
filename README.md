# Fox-Li Laser Cavity Simulator

This repository houses a high-performance, GPU-accelerated numerical testbed for simulating the steady-state transverse electromagnetic modes of open laser resonators. Built natively in Python using the JAX framework, this simulator bypasses the computational bottlenecks of standard iterative solvers. By leveraging JIT compilation and XLA optimization, it processes dense optical field arrays through complex round-trip propagations in milliseconds.

### Project Overview
Unlike standard paraxial models, this physics engine utilizes the exact Angular Spectrum Method (ASM) for free-space propagation. It is specifically architected to natively handle complex spatial distortions and rigorous cavity physics, including:

* **Dynamic Optical Misalignment:** Simulates physical spatial decentration and angular tilt in real-time.
* **Non-Uniform Gain Profiling:** Incorporates 2D spatial gain maps to physically model active media.
* **Automated Far-Field Metrics:** Computes the outcoupled near-field intensity, steady-state phase maps, and the far-field diffraction pattern to dynamically calculate the $M^2$ beam quality factor against a fundamental Gaussian reference.

This codebase couples the core JAX-accelerated solver with a custom PyQt5 graphical interface, allowing for the real-time visualization and dynamic manipulation of complex resonator setups. 

### Authors
* **Debanjan Halder**
  Project Scientist, Computational Optics Laboratory, Optics & Photonics Centre (OPC)
  Indian Institute of Technology, Delhi

* **Prof. Kedar Khare**
  Professor and Head, Optics & Photonics Centre (OPC)
  Head, Centre for Sensors, Instrumentation and Cyber Physical System Engineering (CeNSE)
  Indian Institute of Technology, Delhi
