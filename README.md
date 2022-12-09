# Data Cube Backend

Simple [FastAPI](https://fastapi.tiangolo.com/) backend to access metadata and subsets of n-dimensional datasets using [xarray](https://docs.xarray.dev/en/stable/).

## Setup

- setup Python environment with conda: `conda env create --file environment.yml`
- activate the environment: `conda activate data-cube-backend`
- move or symlink xarray-compatible datasets to the folder `data`
- start the server: `uvicorn backend:app`
  - add `--reload` during development for automatic reloading
