[Wavemaker parameter file]
To create the waveform using the wavemaker parameter file, place the parameter file (seldec_onres_pm1p5.par) in the ./exp/stan/nmr/wavemaker/par/ directory under the Topspin directory. Then, type the following command on Topspin:
wvm -f seldec_onres_pm1p5.par
This will create the waveform (seldec_onres_pm1p5) in the ./exp/stan/nmr/lists/user/ directory under the Topspin directory.
See the Wavemaker manual for details; the procedure may vary depending on the version of Topspin.

[Example settings for selective decoupling]
(1) Set cpdprg2 to "onepulse.sp23"
(2) Set SPNAM23 to "seldec_onres_pm1p5.wv"
(3) Set p23 to 40000 [us].
(4) Set the power level spw22 to the value corresponding to the 734.6 us, 90 deg pulse of 1H.
