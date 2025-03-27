# Bayesian Optimization

This repository contains some basic code for understanding and applying 
Bayesian Optimization, as well as a bonus notebook on the basics of GPyTorch.

Check out the [notebooks/](notebooks/) folder for:
1. An example notebook showcasing the basics of using GPyTorch.
2. An example notebook showcasing the basics of using Ax.

Check ou the [templates/](templates/) folder for:
1. A boiler plate code for applying Bayesian Optimization using [Ax](https://ax.dev/).

The requirements for using this code are outlined [here](https://ax.dev/docs/installation). In brief, you should use Conda for Pytorch, but pip for everything else.

My recommendation is:
```bash
conda create -n ax-env
conda activate ax-env
conda install pytorch torchvision -c pytorch 
pip install ax-platform
```