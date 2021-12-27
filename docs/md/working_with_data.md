---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Working with data

Regardless of rather we have real or simulated data, the main interface for `whimstan` to x-ray data is via the `Database` object which we can use to both examine the data as well as ship the data off to Stan. 

First, let's examine the simulated data we created in the previous section.


```python
%matplotlib inline
from jupyterthemes import jtplot
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)

from whimstan import Database

```

```python
db = Database.read("data.h5")
```

<!-- #region -->
## The catalog object

The database contains a catalog object that holds info about the observations. If the database is made from simulations, the catalog object will also know about the simulated info for each GRB.


We can access the catalog dictionary with the names of the GRBs.
<!-- #endregion -->

```python
db.catalog.catalog['000']
```

```python
db.catalog.catalog['001']
```

<!-- #region -->
## Getting 3ML plugins from the database


We can create plugins from the data stored in the database for e.g. fitting the data in 3ML:
<!-- #endregion -->

```python
p = db.plugins['000']
```

```python
p.view_count_spectrum();
```

```python
p = db.plugins['001']
p.view_count_spectrum();
```

<!-- #region -->
## Getting an 3ML analysis from the database


Additionally, we can build an analysis from the database.
<!-- #endregion -->

```python
ba = db.build_3ml_analysis(id=0)
```

Now we can fit in 3ML!

