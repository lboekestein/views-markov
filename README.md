<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://github.com/user-attachments/assets/1ec9e217-508d-4b10-a41a-08dface269c7" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>


# **VIEWS-Markov**: A Markov-based Model for Conflict Forecasting

> **Part of the [VIEWS Platform](https://github.com/views-platform) ecosystem for large-scale conflict forecasting.**

## 📚 Table of Contents  

1. [Overview](#overview)
2. [Structure of repository](#structure-of-repository)
3. [Known issues and TODOs](#known-issues-and-todos)

<!-- 2. [Role in the VIEWS Pipeline](#role-in-the-views-pipeline)  
3. [Features](#features)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Architecture](#architecture)  
7. [Project Structure](#project-structure)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [Acknowledgements](#acknowledgements)   -->


## Overview

**MarkovModel** Is a Python implementation of a Markov-based model for conflict forecasting. It predicts the number of conflict-related fatalities in a given month in two steps 
1. It first computes the probability of 4 different Markov states (War, Peace, Escalation, De-escalation) based on the Markov state of the current month and a number of covariates. 
2. It then uses the predicted probabilities of the Markov states to compute a weighted average of the expected number of fatalities in each state, which gives the final prediction of the number of fatalities for the target month.

The implementation in this repository is a translation of an existing implementation of Markov models in R. The original R code can be found in the [`r_version/`](sandbox/r_version/old_pipeline_version/) directory, or in the [viewsforecasting repository](https://github.com/prio-data/viewsforecasting/tree/main/Tools/new_markov). 

## Structure of repository

- [`views_markov/`](views_markov/): Contains the source code for the Markov model implementation in Python.
    - [`markov_model.py`](views_markov/model/markov_model.py): The main module implementing the Markov model.
    - [`markovmodel_manager.py`](views_markov/manager/markovmodel_manager.py): A manager class for training and evaluating the Markov model.
- [`r_version/`](sandbox/r_version/old_pipeline_version/): Contains the original R implementation of the Markov model.
- [`notebooks/`](sandbox/notebooks/): Contains Jupyter notebooks for experimentation and testing of Markov models.

## Known issues and TODOs

- Currently the MarkovModel class does not support PGM-level data (i.e., it assumes country-level data only).
    - Index columns are hardcoded to `["country_id", "month_id"]`, which may not be appropriate for PGM-level data. See the __init__() method in `class MarkovModel` for details.
- The computation of Markov states is currently down as a non-vectorized row-wise operation, which is inefficient for large datasets. This is done in the `_add_markov_states()` method. A more efficient vectorized implementation should be developed.