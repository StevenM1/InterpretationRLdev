# Interpretation of individual differences in computational neuroscience using the latent input approach

This repository contains R code to simulate scenarios from the paper "Interpretation 
of individual differences in computational neuroscience using the latent input approach". 
Code is written by Steven Miletic and Jessica Schaaf. Although code is now specified for
reinforcement learning examples, it can be adapted to test the scenarios for other processes,
tasks and models.

Basic functions:
- fMRI.R contains R code to simulate fMRI data (incl. the generation of design matrices).
- simulation_functions.R contains basic simulation functions for simulating and fitting RL data and accompanying neural data.

Specific scenarios:

- scenarios_1_3_7.R contains code that allows one to simulate RL and fMRI data using different learning rates that either co-occur (scenario 3) or not (scenario 1) with differences in to neural coding parameter ('phi' in the paper, 'beta_PE' in the code -- the terminology is a bit unfortunate since both the inverse temperature parameter and GLM parameters are commonly referred to as 'beta'). It also contains a model fitting part to investigate what happens to the neural coding parameter when one incorrectly assumes no individual differences in the learning rate (scenario 7).

- scenario_4.R contains simulation code to show that ignoring individual differences in the
duration of the neural response lead to spurious individual differences in the neural coding parameter.

- scenarios_5_6.R contains code that covers the effects of differences in inverse temperatures and outcome sensitivity.

- scenario_7_misspecification.R contains code that covers the model misspecification case of scenario 7. Note that this code slightly adapted some of the functions defined in simulation_functions.R; hence they're re-defined in there.
