# GCpro

Use prospector to fit globular cluster photometry and uncalibrated spectra

## Install

Follow the steps in `install.sh`, with appropriate changes to path names

## Use

Now you can either fire up the notebook

```sh
conda activate pro
jupyter notebook GCSpecPhotDemo.ipynb
```

or use the supplied parameter file for command-line operation (in the same directory as the `gc1.*.dat` data files)

```sh
conda activate pro
python gc_specphot_params.py --object_name gc1 --outfile gc1_dynesty --luminosity_distance 10 \
                             --remove_spec_continuum --set_lsf \
                             --dynesty --nested_method rwalk --nested_posterior_thresh 0.05
```

You can see all the possible options with

```sh
conda activate pro
python gc_specphot_params.py --help
```
