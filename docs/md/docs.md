---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Intro

This is a simple tool to allow scripting of the awesome [XRT website](https://www.swift.ac.uk/xrt_products/index.php). The official API (which does a lot more) can be found [here](https://www.swift.ac.uk/user_objects/API/). 

These tools extend the functionality a bit. 

```python
%matplotlib inline

from xrt_spec_dl import XRTLightCurve, download_xrt_spectral_data

```

## Downloading Spectra

This allows you to download time-sliced spectra for spectral analysis. You can choose PC or WT mode.

```python
obs_id = "01071993"

grb = "210905A"
```

```python
download_xrt_spectral_data(obs_id=obs_id,
    name=f"GRB {grb}",
    mode="PC",
    tstart=239,
    tstop=446,
    destination_dir=".")


!ls

```

## Downloading Light curves

To get the light curve data for plotting purposes, we need to know the obs_id of the GRB. the class will look at the Swift data online and pull it for making plots. In the future, a cache options will be included for off-line use. 


```python
lc = XRTLightCurve(obs_id=obs_id)
```

The WT and PC data exist as astropy tables.

```python
lc.wt_data
```

```python
lc.pc_data
```

Finally, we can plot the data

```python
lc.plot(pc_mode=True, wt_mode=True);
```

```python

```
