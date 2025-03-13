# SAO_k-AUS
Improved Snow Ablation Optimization for Multilevel Threshold Image Segmentation

## Abstract
Snow ablation optimization (SAO) is a novel metaheuristic algorithm (MA). However, we observed certain issues in the original SAO, such as poor capacity in escaping from local optima and slow convergence. To address these limitations, we introduce two strategies: the asynchronous update strategy (AUS) and the top-k survival mechanism. We name our proposal SAO_k-AUS. In the original SAO, the segregation of search and update delays the improved information sharing, and AUS integrates update processes following each individual’s search behavior, facilitating superior knowledge from elites. Additionally, the original SAO adopts an all-acceptance selection principle, maintaining diversity but cannot guarantee the solution quality. Thus, we introduce the top-k survival mechanism to ensure the survival of elites. Comprehensive numerical experiments on CEC2013 and CEC2020 benchmark functions, engineering problems, and image segmentation tasks were conducted to evaluate our proposal against eight state-of-the-art MAs. The experimental results and statistical analyses confirm the efficiency of -AUS. Moreover, the ablation experiments investigate the contribution of two strategies, and we recommend using both proposed strategies simultaneously. The source code of this research is made available in https://github.com/RuiZhong961230/SAO_k-AUS.

## Citation
@article{Zhong:24,  
title={Improved Snow Ablation Optimization for Multilevel Threshold Image Segmentation},  
author={Rui Zhong and Chao Zhang and Jun Yu},  
journal={Cluster Computing},  
pages={1-32},  
year={2024},  
publisher={Springer},  
doi = {10.1007/s10586-024-04785-w},  
}


## Datasets and Libraries
CEC benchmarks and Engineering problems are provided by opfunu==1.0.0 and enoppy==0.1.1 libraries, respectively, Deep learning models are provided by the Pytorch==2.1.2 library, and datasets of multilevel image segmentation are provided by the skimage==0.24.0 and opencv-python==4.8.0.74 libraries.

## Contact
If you have any questions, please don't hesitate to contact zhongrui[at]iic.hokudai.ac.jp
