# A computation model of Human Action Understanding


# Overview

This is test code for a Computational Model for Human Action Understanding. Here we have considered two types of actions namely `chasing` and `pushing` in the thematic role understanding scenarios. We proposed some motion-based heuristics to model these two actions. For details, please see our [paper](https://ieeexplore.ieee.org/document/8614142).

# Prerequisites
- python 2.7 or might work for python3.x (not tested)
- numpy
- [matplotlib](https://matplotlib.org/)
- [opencv](https://pypi.org/project/opencv-python/)
- [scikit-learn](https://scikit-learn.org/stable/)

# Description

This code has various dependencies, some are listed in **Prerequisites** section. The code first will generate some video dataset (synthetic) for `chasing` and `pushing` actions. Then test data based on our heuristic-based model.   

Please see the **Example** section for different parameter combination of two actions. 

# Examples:

To see different parameters option, from a shell (Bash, Bourne, or else) run:

```bash
python2.7 ss_action_data_classification.py -h
```

To test the model on default parameters values (say you want to test the model for `'num_video_files=30'` videos for each action), from a shell (Bash, Bourne, or else) run the following script:

```bash
python2.7 ss_action_data_classification.py --num_video_files 30
```
