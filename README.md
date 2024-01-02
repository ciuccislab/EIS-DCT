# EIS-DCT

Electrochemical impedance spectroscopy (EIS) is an experimental technique widely used in to analyze the working principles of batteries, fuel cells, and supercapacitors. While the measurement procedure is relatively easy, the analysis of the data collected remains difficult. The non-parametric distribution of relaxation times (DRT) has recently gained a broader acceptance due to its ability to separate electrochemical phenomena and compute physico-chemical parameters. However, the DRT is unsuitable to analyze the low-frequency part of the impedance when the imaginary component of the impedance diverges, as it is the case for systems characterized by blocking electrodes. To overcome this limitation, another distribution, namely the distribution of capacitive times (DCT), can be used for the admittance, i.e., the inverse of the impedance. In this article, we derive the theoretical framework of the DCT, highlight its advantages compared to the DRT, and propose a new method for deconvolving it from measured impedances. 


![Baptiste_V9](https://github.com/ciuccislab/EIS-DCT/assets/57649983/12f90a85-7af4-473d-8e04-ddd97a4a705c)




# Dependencies
numpy

scipy

matplotlib

pandas

# Tutorials
1. **ex1_single ZARC Model.ipynb**: this notebook gives detail procedure of how to recover the DRT from the impedance generated using a single ZARC model consisting of a resistance placed in parallel to a constant phase element (CPE) The frequency range is from 1E-4 Hz to 1E4 Hz with 10 points per decade (ppd).
2. **ex2_double ZARC Model.ipynb** : this notebook demonstrates how the fGP-DRT can capture overlapping timescales with two ZARC models arranged in series. The frequency range is from 1E-4 Hz to 1E4 Hz with 10 ppd.
3. **ex3_single_ZARC_plus_an_inductor.pynb** : this notebook adds an inductor to the model used in "**example1_single ZARC Model.ipynb**"
4. **ex4_BLF_pO2_60percent_Temp_500_C.ipynb** : this notebook displays the DRT analysis of the BLF impedance spectra from fuel cell. The real experimental EIS data is read from a csv file, the DRT is predicted by the fGP-DRT model, the complete impedance is, therefore, recovered and compared with the equivalent circuit model (ECM) consisting of two ZARCs
5. **ex5_SCFN_3percent_H2O_Temp_500_C.ipynb** : this notebook shows the DRT analysis of the SCFN impedance spectra from real experiment. Also the real EIS data is read from a csv file, the DRT is predicted by the fGP-DRT model, the real and imaginary components of the impedance are recovered and compared with the 2ZARCs ECM. 

# Citation

```
@article{maradesa2022probabilistic,
  title={The Probabilistic Deconvolution of the Distribution of Relaxation Times with Finite Gaussian Processes},
  author={Maradesa, Adeleke and Py, Baptiste and Quattrocchi, Emanuele and Ciucci, Francesco},
  journal={Electrochimica Acta},
  pages={140119},
  year={2022},
  publisher={Elsevier}
}

```

# References
[1] Maradesa, A., Py, B., Quattrocchi, E., & Ciucci, F. (2022). The probabilistic deconvolution of the distribution of relaxation times with finite Gaussian processes. Electrochimica Acta, 413, 140119. https://doi.org/10.1016/j.electacta.2022.140119.

[2] Ciucci, F. (2018). Modeling electrochemical impedance spectroscopy. Current Opinion in Electrochemistry.132-139. https://doi.org/10.1016/j.coelec.2018.12.003. 

[3] Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools. Electrochimica Acta, 184, 483-499. https://doi.org/10.1016/j.electacta.2015.09.097.

[4] Saccoccio, M., Wan, T. H., Chen, C., & Ciucci, F. (2014). Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods-a theoretical and experimental study. Electrochimica Acta, 147, 470-482. https://doi.org/10.1016/j.electacta.2014.09.058.

[5] Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439-454. https://doi.org/10.1016/j.electacta.2015.03.123.

[6] Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. Electrochimica Acta, 247, 1117-1129. https://doi.org/10.1016/j.electacta.2017.07.050.

[7]   Liu, J., & Ciucci, F. (2020). The deep-prior distribution of relaxation times. Journal of The Electrochemical Society, 167 (2) 026-506. https://doi.org/10.1149/1945-7111/ab631a

[8] Liu, J., Wan, T. H., & Ciucci, F.(2020). A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357, 136-864. https://doi.org/10.1016/j.electacta.2020.136864.

[9] Liu, J., & Ciucci, F. (2020). The Gaussian process distribution of relaxation times: A machine learning tool for the analysis and prediction of electrochemical impedance spectroscopy data. Electrochimica Acta, 135316. https://doi.org/10.1016/j.electacta.2019.135316.

[10] Quattrocchi, E., Wan, T. H., Belotti, A., Kim, D., Pepe, S., Kalinin, S. V., Ahmadi, M., and Ciucci, F. (2021). The deep-DRT: A deep neural network approach to deconvolve the distribution of relaxation times from multidimensional electrochemical impedance spectroscopy data. Electrochimica Acta, 139010. https://doi.org/10.1016/j.electacta.2021.139010
