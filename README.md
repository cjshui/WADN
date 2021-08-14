# WADN
Wasserstein Aggregation Domain Network
A pytorch implementation of [Aggregating From Multiple Target-Shifted Sources](https://arxiv.org/abs/2105.04051)



## Prerequisites

- Pytorch >=1.0, Torchvision >=0.2 
- Scikit-learn >= 0.19.1
- CVXPY>=1.9
 
## Models

- 'was_main_labeled.py ': Evaluation with limited target label prediction
- 'was_main_uda.py': Code for Unsupervised DA
- 'solver.py' Solver for estimating the optimal weights and label distribution ratio
## How to cite

```xml

@InProceedings{pmlr-v139-shui21a,
  title = 	 {Aggregating From Multiple Target-Shifted Sources},
  author =       {Shui, Changjian and Li, Zijian and Li, Jiaqi and Gagn{\'e}, Christian and Ling, Charles X and Wang, Boyu},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {9638--9648},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/shui21a/shui21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/shui21a.html}
}

```

