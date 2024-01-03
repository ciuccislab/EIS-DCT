# Project

## EIS-DCT: Unlocking the Distributive of Capacitive Times in Electrochemical Impedance Spectroscopy

This repository contains some of the source code used for the paper titled *Theory to Practice: Unlocking the Distributive of Capacitive Times in Electrochemical Impedance Spectroscopy*. Electrochimica Acta, 413, 143741. https://doi.org/10.1016/j.electacta.2023.143741. The article is available online at [Link](https://doi.org/10.1016/j.electacta.2023.143741) and in the [docs](docs) folder. 

Electrochemical impedance spectroscopy (EIS) is an experimental technique widely used to analyze the working principles of batteries, fuel cells, and supercapacitors. While the measurement procedure is relatively easy, the analysis of the data collected remains difficult. The non-parametric distribution of relaxation times (DRT) has recently gained a broader acceptance due to its ability to separate electrochemical phenomena and compute physico-chemical parameters. However, the DRT is unsuitable to analyze the low-frequency part of the impedance when the imaginary component of the impedance diverges, as it is the case for systems characterized by blocking electrodes. To overcome this limitation, another distribution, namely the distribution of capacitive times (DCT), can be used for the admittance, i.e., the inverse of the impedance. In this article, we derive the theoretical framework of the DCT, highlight its advantages compared to the DRT, and propose a new method for deconvolving it from measured impedances. 


![Baptiste_V9](https://github.com/ciuccislab/EIS-DCT/assets/57649983/12f90a85-7af4-473d-8e04-ddd97a4a705c)




# Dependencies
numpy

scipy

matplotlib

pandas

PyTorch

# Tutorials
1. **ex1_2ZARC+Warburg.ipynb**: this notebook shows how to deconvolve the DCT of synthetic EIS data generated using the series association of a double ZARC element and a generalized Warburg element;
2. **ex2_deLevie.ipynb** : this notebook shows how to deconvolve the DCT of synthetic EIS data generated with de Levie model;
3. **ex3_ML621.ipynb** : this notebook shows how to deconvolve the DCT of real EIS data measured by our group on a Panasonic ML621 battery at the states of charge 10%, 20%,..., 100%.

# Citation

```
@article{py2023theory,
  title={Theory to Practice: Unlocking the Distribution of Capacitive Times in Electrochemical Impedance Spectroscopy},
  author={Py, Baptiste and Maradesa, Adeleke and Ciucci, Francesco},
  journal={Electrochimica Acta},
  pages={143741},
  year={2023},
  publisher={Elsevier}
}

```

# References
[1] T.H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools, Electrochim. Acta. 184 (2015) 483-499. https://doi.org/10.1016/j.electacta.2015.09.097.

[2] A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, Selecting the regularization parameter in the distribution of relaxation times, J. Electrochem. Society. 170-3 (2023) 030502. https://doi.org/10.1149/1945-7111/acbca4 

[3] M. Saccoccio, T.H. Wan, C. Chen, F. Ciucci, Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods-a theoretical and experimental study, Electrochim. Acta. 147 (2014) 470-482. https://doi.org/10.1016/j.electacta.2014.09.058.

[4] J. Liu, F. Ciucci, The deep-prior distribution of relaxation times, J. Electrochem. Society. 167-2 (2020) 026506. https://doi.org/10.1149/1945-7111/ab631a.

[5] E. Quattrocchi, T.H. Wan, A. Belotti, D. Kim, S. Pepe, S.V. Kalinin, M. Ahmadi, F. Ciucci, The deep-DRT: A deep neural network approach to deconvolve the distribution of relaxation times from multidimensional electrochemical impedance spectroscopy data, Electrochim. Acta. 392 (2021) 139010. https://doi.org/10.1016/j.electacta.2021.139010.

[6] E. Quattrocchi, B. Py, A. Maradesa, Q. Meyer, C. Zhao, F. Ciucci, Deconvolution of electrochemical impedance spectroscopy data using the deep-neural-network-enhanced distribution of relaxation times, Electrochim. Acta. 439 (2023) 141499. https://doi.org/10.1016/j.electacta.2022.140119.

[7] K. Yang, J. Liu, Y. Wang, X. Shi, J. Wang, Q. Lu, F. Ciucci, Z. Yang, Machine-learning-assisted prediction of long-term performance degradation on solid oxide fuel cell cathodes induced by chromium poisoning, J. Materials Chem. A. 10-44 (2022) 23683-23690. https://doi.org/10.1039/D2TA03944C.

[8] F. Ciucci, Modeling electrochemical impedance spectroscopy. C. Opin. Electrochem., 13 (2019) 132-139. https://doi.org/10.1016/j.coelec.2018.12.003.

