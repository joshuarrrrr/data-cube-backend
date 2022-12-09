from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import xarray as xr
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# use the local data folder as a source for datasets
datasets = {path.stem: xr.open_dataset(path) for path in Path("data").glob("*.nc")}


def is_datetime(arr: xr.DataArray) -> bool:
    return np.issubdtype(arr.dtype, np.datetime64)


def get_array_values(arr: xr.DataArray) -> list[float] | list[int] | list[str]:
    if is_datetime(arr):
        return [date.isoformat() for date in pd.to_datetime(arr, utc=True)]
    return arr.values.tolist()


def get_variable_info(arr: xr.DataArray) -> dict:
    return {
        "dimensions": arr.dims,
        "shape": arr.shape,
    }


def get_dataset_info(ds: xr.Dataset) -> dict:
    return {
        "dimensions": list(ds.dims),
        "coordinates": {coord: get_array_values(ds[coord]) for coord in ds.coords},
        "variables": {
            data_var: get_variable_info(ds[data_var]) for data_var in ds.data_vars
        },
    }


def check_has_dataset(dataset: str):
    if not dataset in datasets:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset}' not found!",
        )


def check_has_variable(dataset: str, variable: str):
    check_has_dataset(dataset)
    if not variable in datasets[dataset].data_vars:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset}' has no variable named '{variable}'!",
        )


@app.get("/datasets")
def get_all_datasets() -> dict:
    return {name: get_dataset_info(ds) for name, ds in datasets.items()}


@app.get("/datasets/{dataset}")
def get_dataset(dataset: str) -> dict:
    check_has_dataset(dataset)
    return get_dataset_info(datasets[dataset])


@app.get("/datasets/{dataset}/variables")
def get_variables(dataset: str) -> dict:
    check_has_dataset(dataset)
    ds = datasets[dataset]
    return {variable: get_variable_info(ds[variable]) for variable in ds.data_vars}


@app.get("/datasets/{dataset}/variables/{variable}")
def get_variable(dataset: str, variable: str) -> dict:
    check_has_variable(dataset, variable)
    return get_variable_info(datasets[dataset][variable])


@app.get("/datasets/{dataset}/variables/{variable}/data")
def get_data_dice(dataset: str, variable: str, request: Request) -> Response:
    check_has_variable(dataset, variable)
    arr = datasets[dataset][variable]

    # collect all query parameters that match a dimension of the array
    dim_params = {
        dim: request.query_params.getlist(dim)
        for dim in arr.dims
        if dim in request.query_params
    }

    # create a range dictionary
    indexers = {
        dim: slice(*values)  # use slice with two values if two values are given
        if len(values) == 2
        else values[0]  # use the single value if there is exactly one
        if len(values) == 1
        else values  # otherwise use the full list
        for dim, values in dim_params.items()
    }

    # convert to pandas DataFrame and return serialized JSON
    df = arr.sel(indexers).to_dataframe()
    return Response(content=df.to_json(orient="table"), media_type="application/json")
