#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""make_data.py - script for making fake GC data.
"""

import numpy as np

from sedpy.observate import load_filters
from gc_specphot_params import build_sps, build_model

np.random.seed(101)
phot_snr = 50
spec_snr = 10
objname = "gc1"

sps = build_sps()
model = build_model()
filternames = ["galex_NUV", "sdss_g0", "sdss_i0", "twomass_Ks"]
filters = load_filters(filternames)
obs = dict(filters=filters, wavelength=np.linspace(4000, 7000, 3001), spectrum=None, maggies=None)

model.params["mass"] = np.array([1e8])
model.params["tage"] = np.array([4.])
model.params["sigma_smooth"] = 40.
model.params["lumdist"] = np.array([10.])

spec, phot, mfrac = model.predict(model.theta, obs=obs, sps=sps)


conv = 3631e3  # maggies to mJy
with open("{}.phot.dat".format(objname), "w") as out:
    out.write("filter_name   flux(mJy)   unc(mJy)\n")
    for i, f in enumerate(obs["filters"]):
        unc = phot[i] / phot_snr
        flux = phot[i] + unc * np.random.normal()
        out.write("{:12}  {:6.4f}      {:6.4f}\n".format(f.name, flux*conv, unc*conv))

with open("{}.spec.dat".format(objname), "w") as out:
    out.write("wavelength(AA)   flux(mJy)   unc(mJy)\n")
    unc = spec / spec_snr
    flux = spec + unc * np.random.normal(size=len(spec))
    for i, w in enumerate(obs["wavelength"]):
        out.write("{:6.3f}     {:6.4f}   {:6.4f}\n".format(w, flux[i]*conv, unc[i]*conv))
