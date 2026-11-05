import xarray as xr


ds = xr.open_dataset('dustscenario_MY36_v2-5.nc')
print(ds)