# PedestrianQuantumSimulation

## Introduction
PedestrianQuantumSimulation is a quantum-inspired pedestrian movement modeling framework. 
The model represents pedestrian distribution as a probability field by constructing 
an urban potential surface (V_total) composed of:

- V_urbs – Built environment effects
- V_civitas – Social / pedestrian interaction effects

The total potential is then used to generate pedestrian probability maps using:
- A quantum-inspired Schrödinger-based solver
- A classical Boltzmann baseline model

---

## Data Requirements

The model requires raster-based GIS inputs (GeoTIFF format) with identical resolution and extent.

Typical inputs include:

- building_density.tif
- street_centrality.tif
- building_height.tif
- pedestrian_density.tif
- poi_density.tif
- wall_constraint.tif
- mean_depth.tif
- isovist.tif
- vehicle_density.tif
- pedestrian_accessibility.tif

Optional validation datasets:
- observed_pedestrian_morning.tif
- observed_pedestrian_afternoon.tif
- observed_pedestrian_evening.tif

Outputs:
- V_total_*.tif
- pedestrian_probability_*.tif
- PNG visualizations and animations

---

## Code Structure

Main Components:

- main.py – CLI entry point
- enhanced_model.py – Core modeling pipeline
- quantum_coefficient_optimizer.py – Coefficient optimization
- model_validation.py – Model validation and sensitivity analysis
- urban_scenario_simulator.py – Scenario-based simulation

Configuration:
- config.yaml – Defines parameters, coefficients, quantum settings, and data paths

---

## Citing

If you use this model in academic work, please cite:

Rajapaksha, S. D. (2026). PedestrianQuantumSimulation. GitHub repository.
