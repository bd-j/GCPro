#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, time
import numpy as np

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

VERS = sys.version_info[0]

# --------------
# Observational Data
# --------------

def build_obs(object_name="gc1", **extras):
    """Build a dictionary of observational data.  In this example
    we will read from text files named <object_name>.spec.dat and <object_name>.phot.dat

    Parameters
    ----------
    object_name : string
        The name of the object, used to find data files

    Returns
    -------
    obs : dictionary
        A dictionary of observational data to use in the fit.
    """
    from prospect.utils.obsutils import fix_obs
    from sedpy.observate import load_filters

    # The obs dictionary, empty for now
    obs = {}

    # These are the names of the relevant filters,
    # in the same order as the photometric data (see below)
    phot_file = "{}.phot.dat".format(object_name)
    spec_file = "{}.spec.dat".format(object_name)

    # --- Photometric Data ---

    # this is the description of the column formats
    if VERS == 2:
        stype = "S20"
    elif VERS == 3:
        stype = "U20"
    cols = [("filtername", stype), ("flux", np.float), ("unc", np.float)]
    # read the file
    phot = np.genfromtxt(phot_file, dtype=np.dtype(cols), skip_header=1)
    # And here we instantiate the `Filter()` objects using methods in `sedpy`,
    # and put the resulting list of Filter objects in the "filters" key of the `obs` dictionary
    # These Filter objects will perform projections of the model SED onto the filters.
    obs["filters"] = load_filters(phot["filtername"])
    # Now we store the measured fluxes for a single object, **in the same order as "filters"**
    # The units of the fluxes need to be maggies (Jy/3631) so we will do the conversion here too.
    obs["maggies"] = phot["flux"] * 1e-3 / 3631
    # And now we store the uncertainties (again in units of maggies)
    obs["maggies_unc"] = phot["unc"] * 1e-3 / 3631
    # Now we need a mask, which says which flux values to consider in the likelihood.
    # IMPORTANT: the mask is *True* for values that you *want* to fit,
    # and *False* for values you want to ignore.  Here we ignore the spitzer bands.
    obs["phot_mask"] = np.array(['spitzer' not in f.name for f in obs["filters"]])
    # This is an array of effective wavelengths for each of the filters.
    # It is not necessary, but it can be useful for plotting so we store it here as a convenience
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

    # --- Spectroscopic Data ---

    # this is the description of the column formats
    cols = [("wavelength", np.float), ("flux", np.float), ("unc", np.float)]
    # read the file
    spec = np.genfromtxt(spec_file, dtype=np.dtype(cols), skip_header=1)
    # add the wavelengths (restframe, vacuum)
    obs["wavelength"] = spec["wavelength"]
    # (this would be the spectrum in units of maggies)
    obs["spectrum"] = spec["flux"] * 1e-3 / 3631
    # (spectral uncertainties are given here)
    obs['unc'] = spec["unc"] * 1e-3 / 3631
    # (again, to ignore a particular wavelength set the value of the
    #  corresponding elemnt of the mask to *False*)
    obs['mask'] = spec["unc"] > 0

    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    # That's it!
    return obs


def build_sps(zcontinuous=1, **extras):
    """
    Parameters
    ----------
    zcontinuous : int or bool
        A value of 1 insures that we use interpolation between SSPs to
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    from prospect.sources import FastSSPBasis
    sps = FastSSPBasis(zcontinuous=zcontinuous)
    return sps


def build_model(object_redshift=None, luminosity_distance=0.0, fixed_metallicity=None,
                remove_spec_continuum=True, add_neb=False, **extras):
    """Build a prospect.models.SedModel object

    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate
        for this redshift. Otherwise, the redshift will be zero.

    :param luminosity_distance: (optional, default: 10)
        The luminosity distance (in Mpc) for the model.  Spectra and observed
        frame (apparent) photometry will be appropriate for this luminosity distance.

    :param fixed_metallicity: (optional, default: None)
        If given, fix the model metallicity (:math:`log(Z/Z_sun)`) to the given value.

    :param add_duste: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to the model.

    :returns model:
        An instance of prospect.models.SedModel
    """
    from prospect.models.sedmodel import PolySpecModel, PolySedModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["ssp"]

    # --- Adjust Model Components ---
    # Here we add to and tweak the model components to get something that can describe our data

    # Fit for velocity dispersion.  This adds parameters necessary for fitting for spectral broadening
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=5.0, maxi=100.0)

    # This removes the continuum from the spectroscopy. Highly recommended
    # when modeling both photometry & spectroscopy
    # If the normalization between the spectrum & photometry is off by ~> 10
    # orders of magnitude, this can get numerically unstable.
    if remove_spec_continuum:
        model_params.update(TemplateLibrary['optimize_speccal'])

        # Here we set the order of the polynomial used to normalize the observed spectrum to the model.
        # Avoid using very high order polynomials as they will remove absorption lines,
        # and beware that spectral breaks (e.g. dn4000) may be optimized out by this procedure.
        # Rule of thumb: this removes continuum shape on wavelength scales
        # of (total_wavelength_range/polyorder).
        model_params['polyorder']["init"] = 12

    # Now add the lumdist parameter by hand as another entry in the dictionary.
    # This will control the distance since the redshift is possibly decoupled from the hubble flow.
    # If luminosity_distance is less than or equal to 0,
    # then the distance is controlled by the "zred" parameter and a WMAP9 cosmology.
    if luminosity_distance > 0:
        model_params["lumdist"] = {"N": 1, "isfree": False,
                                   "init": luminosity_distance, "units": "Mpc"}

    # if we specify a redshift, let the model know, otherwise fit for it
    if object_redshift is not None:
        # make sure zred is fixed
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift
    else:
        model_params["zred"]['init'] = 0.0
        model_params["zred"]['isfree'] = True

    # Change the model parameter specifications based on some keyword arguments
    # Here, if a value is passed in the 'fixed_metallicity' keyword,
    # we fix the metallicity in units of log(Z/Zsun) to that value
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    # This adds nebular emission using a fixed CLOUDY grid.
    # The gas-phase metallicity is set to the stellar metallicity by default
    # and the ionization parameter is fixed by default.
    # If there are nebular lines in the spectrum which is either (a) off of the CLOUDY grid,
    # or (b) not caused by star formation, recommend using analytical marginalization instead.
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])

    # --- Priors ---

    # Here we adjust our priors
    model_params["mass"]["prior"]    = priors.LogUniform(mini=1e4, maxi=1e10)
    model_params["dust2"]["prior"]   = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["tage"]["prior"]    = priors.TopHat(mini=0.1, maxi=13.7)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.98, maxi=0.19)
    model_params["zred"]["prior"]    = priors.TopHat(mini=-0.05, maxi=0.05)

    # --- Initial Values ---
    # These are not really necessary, but they make the plots look better
    model_params["mass"]["init"] = 1e8
    model_params["sigma_smooth"]["init"] = 20.0

    # Now instantiate the model object using this dictionary of parameter specifications
    # Could also use SedModel, but SedModel is old and tired, SpecModel is the new hotness
    model = PolySpecModel(model_params)

    return model


# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None


# -----------
# Everything
# ------------

def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__ == '__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--object_name', type=str, default="gc1",
                        help="Filename.")
    parser.add_argument('--object_redshift', type=float, default=0.0,
                        help=("Redshift for the model"))
    parser.add_argument('--luminosity_distance', type=float, default=0.0,
                        help=("Luminosity distance in Mpc. Defaults to 10pc "
                              "(for case of absolute mags)"))
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--remove_spec_continuum', action="store_true",
                        help="Whether to optimize out a continuum polynomial")

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    print(model)

    if args.debug:
        sys.exit()

    # Name your output file. Here it takes the 'outfile' keyword
    hfile = "{0}_{1}_mcmc.h5".format(args.outfile, int(time.time()))
    output = fit_model(obs, model, sps, noise, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass