This repository contains the implementation of a hybrid Guidance and Control (G&C) framework that integrates Deep Reinforcement Learning (DRL) with Model Predictive Control (MPC) for spacecraft rendezvous and docking. The DRL agent acts as an "expert tuner," dynamically optimizing MPC parameters in real-time to balance fuel efficiency and computational load in Highly Elliptical Orbits (HEO).

🚀 Key Results
Fuel Savings: 10% to 40% improvement over static MPC.
Efficiency: ~65% reduction in relative computational time.
Safety: 100% mission success rate maintained via MPC constraint enforcement.

📂 Repository Structure
The project is organized into four main stages, following the research workflow:
01-Classical MPC Controller: A standard linear MPC implementation used as a benchmark to evaluate the performance gains of the RL-tuned versions.
02-RL on Th and dt - Horizon & Timestep Tuning: Focuses on training a DRL agent to dynamically adjust the Prediction Horizon ($T_h$) and Control Timestep ($dt$) to optimize computational resources.
03-RL on QR - Weighting Matrix Tuning: Dedicated to training the DRL agent to tune the State (Q) and Control (R) weighting matrices to optimize trajectory tracking and fuel consumption.
04-Testing and Validation - Final Evaluation: Includes Monte Carlo simulations across various orbital conditions to compare the Classical MPC against the Tuned RL-MPC, including performance bar charts and statistical validation.

📖 Documentation
Detailed theoretical background and mission analysis can be found in the Docs/ folder:
Executive Summary: A high-level overview of the methodology and results.
Thesis (Full Text): "Autonomous spacecraft proximity operations in Highly Elliptical Orbits" by Pietro Cavalletti (2025).