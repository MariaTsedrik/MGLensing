# MGlensing

Repository containing a likelihood (based on [this pipeline](https://github.com/Sefa76/photometric_fofR/tree/main)) for photometric probes with [emulated nonlinear power spectra](https://github.com/nebblu/ReACT-emus?tab=readme-ov-file) and various other emulators.


The purpose of this code:
* Perform a quick and light MCMC analysis for Stage IV surveys, which is useful for forecasts, scale-cuts, and studies of projection effects.
* Without inclusion of systematic effects on large scales, we focus on systematics on small/nonlinear scales for standard and extended cosmologies: nonlinear galaxy bias expansion, nonlinearities in the matter power spectrum, baryons.
* Quick analysis of N-body simulations.

This code is not complete to perform a real data analysis (yet), but is a fairly realistic and good-enough in-between solution.



## Available Models
- BACCO-emulator and HMcode2020-emulator: 
    - $\Lambda$ CDM, $w$ CDM and $w_0w_a$ CDM with neutrinos.
- ReACT-based emulators: 
    - normal branch of [DGP gravity](https://arxiv.org/abs/hep-th/0005016) (nDGP); 
    - [Growth index parametrisation](https://arxiv.org/abs/astro-ph/0507263) and [Time-dependent growth index](https://arxiv.org/abs/2304.07281) with Screening;
    - $\mu-\Sigma$ parametrisation with Screening;
    - Interacting Dark Energy also known as [Dark Scattering](https://arxiv.org/abs/1605.05623).
- Galaxy bias:
    - Linear;
    - Phenomenalogical $b(z, k) = b_1(z) + b_2(z) k^2$;
    - Hybrid Lagrangian bias epxansion (HEFT) with BACCO-emulator.
- Baryons:
    - HMcode2020;
    - BCemu;
    - BACCO-emulator.

## How to run

Modify "config.yaml" file and specify
* Observables;
* Specifics of your survey;
* Modelling for your mock synthetic data;
* Modelling for your theoretical predictions;
* File-paths with fiducial data points and priors;
* Output-file name.


You can then run a test likelihood computation:
```python
import MGLensing

MGL = MGLensing.MGL("config.yaml")
MGL.test() 
```

Following files provide examples of using MGL:
* test_plot.py and test_plot_kernels.py : compute and plot observables and modelling components;
* test_mcmc_run.py: run an MCMC chain with [Nautilus sampler](https://github.com/johannesulf/nautilus);
* test_plot_posterior.py: plot posterior distributions with GetDist;
* test_postprocess_S8.py: compute sigma8 and S8 from a chain.

## Documentation
...tbd...


## To-do list for Maria
* new likelihood 


## To-do list for Ottavia
* compare with CosmoSIS and CCL
* photo-z uncertainties 
* sigma8_tot to sigma8_cold
* power-law extrpolation of power spectra

