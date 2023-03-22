# mmd

This repository implements code for magnetic mirror descent, as described in the paper *A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games*.

## Installation
To install the package, run the following code:

```
git clone https://github.com/ssokota/mmd.git
cd mmd
pip install .
```

## Examples

Four example scripts for magnetic mirror descent are included in the examples directory:
1. `examples/nfg/perturbed_rps.py` is a script for computing QREs for perturbed RPS
2. `examples/efg/aqre.py` is a script for computing AQREs in extensive-form games
4. `examples/efg/nash.py` is a script for computing Nash equilibria in extensive-form games
5. `examples/h2h/evaluate.py` is a script for head-to-head evaluation in extensive-form games

There is also an additional example script `examples/efg/cfr.py` for running CFR in extensive-form games.

## Reference

The reference for the paper is:
```
@inproceedings{
sokota2023a,
title={A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games},
author={Samuel Sokota and Ryan D'Orazio and J Zico Kolter and Nicolas Loizou and Marc Lanctot and Ioannis Mitliagkas and Noam Brown and Christian Kroer},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=DpE5UYUQzZH}
}
```

