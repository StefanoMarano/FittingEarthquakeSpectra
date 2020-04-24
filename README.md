## Fitting Earthquake Spectra: Colored Noise and Incomplete Data

This repository stores the code used in the publication 
[Fitting Earthquake Spectra: Colored Noise and Incomplete Data](https://doi.org/10.1785/0120160030) published on the Bulletin of the Seismological Society of America. [How to cite us](#bibtex-entry).

 * [Abstract](#abstract)
 * [Example: Complete data](#example-complete-data)
 * [Example: Incomplete data](#example-incomplete-data)
 * [Executing the code](#executing-the-code)
 * [BibTex Entry](#bibtex-entry)

## Abstract

Spectral analysis of earthquake recordings provides fundamental seismological information. It is used for magnitude calculation, estimation of attenuation, and the determination of fault rupture properties including slip area, stress drop, and radiated energy. Further applications are found in site‐effect studies and for the calibration of simulation and empirically based ground‐motion prediction equations.

We identified two main limitations of the spectral fitting methods currently used in the literature. First, the frequency‐dependent noise level is not properly accounted for. Second, there are no mathematically defensible techniques to fit a parametric spectrum to a seismogram with gaps.

When analyzing an earthquake recording, it is well known that the noise level is not the same at different frequencies, that is, the noise spectrum is colored. The different, frequency‐dependent, noise levels are mainly due to ambient noise and sensor noise. Methods in the literature do not properly account for the presence of colored noise.

Seismograms with gaps are usually discarded due to the lack of methodologies to use them. Modern digital seismograms are occasionally clipped at the arrival of the strongest ground motion. This is also critical in the study of historical earthquakes in which few seismograms are available and gaps are common, significantly decreasing the number of useful records.

In this work, we propose a method to overcome these two limitations. We show that the spectral fitting can be greatly improved and earthquakes with extremely low signal‐to‐noise ratio can be fitted. We show that the impact of gaps on the estimated parameters is minor when a small fraction of the total energy is missing. We also present a strategy to reconstruct the missing portion of the seismogram.

## Example: Complete data

Brune model fit for complete data case, accounting for colored noise. In the three following pictures:
 * Observed seismogram and maximum-likelihood (ML) fit
 * Comparison of the fitted parametric spectrum with the discrete Fourier transform (DFT) of the seismogram
 * Slices of the log-likelihood function, the white cross pinpoints the ML estimate.

![Time-domain fit][FitTime]

![Frequency-domain fit][FitFreq]

![Likelihood function][LogLikelihood]

[FitFreq]: img/Fit_FreqDomain.png "Frequency-domain fit"
[FitTime]: img/Fit_TimeDomain.png "Time-domain fit"
[LogLikelihood]: img/Loglikelihood.png "Likelihood function"

## Example: Incomplete data
Brune model fit for incomplete data case, accounting for colored noise. In the three following pictures:
 * Observed seismogram, unobserved seismogram, time-domain fit, and reconstructed seismogram
 * Comparison of the fitted parametric spectrum with Fourier transform of the complete seismogram and maximum a posteriori (MAP) estimate of the spectrum from observed (incomplete) seismogram
 * Slices of the log-likelihood function, the white cross and the white dot pinpoint the ML estimate for the complete and incomplete data case, respectively. 

![Time-domain fit][iFitTime]

![Frequency-domain fit][iFitFreq]

![Likelihood function][iLogLikelihood]

[iFitFreq]: img/iFit_FreqDomain.png "Frequency-domain fit"
[iFitTime]: img/iFit_TimeDomain.png "Time-domain fit"
[iLogLikelihood]: img/iLoglikelihood.png "Likelihood function"

## Executing the code

 * From command-line, run `python DEMO_CompleteData.py` to fit a complete seismogram. By default, output is saved in the folder `./CompleteDataOutput/`
 * From command-line, run `python DEMO_IncompleteData.py` to fit an incomplete seismogram. By default, output is saved in the folder `./IncompleteDataOutput/`  
 * The input seismogram and other options can be changed within the code.
 * If you do not have Latex installed, set `USETEX=False` in the code. Latex is used for generating nicer picture labels.
 
## BibTex Entry

```
@Article{2017:MarEdwFerFae,
    Title                    = {Fitting Earthquake Spectra: Colored Noise and Incomplete Data},
    Author                   = {Stefano Maran\`o and Ben Edwards and Graziano Ferrari and Donat F\"ah},
    Journal                  = {Bull. Seismol. Soc. Am.},
    Year                     = {2017},
    Month                    = jan,
    Number                   = {1},
    Pages                    = {276--291},
    Volume                   = {107},
    Doi                      = {10.1785/0120160030},
}
```
