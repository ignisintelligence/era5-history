#!/usr/bin/env python3
"""Downloads ERA5's land-sea mask (public, no login) and saves the Australian window as era5_landmask.npz.
Run once; used by finalize.py to keep land points only."""
import os, sys, numpy as np
import era5_daily as e

def main(out='out'):
    import netCDF4
    os.makedirs(out, exist_ok=True)
    keys = [k for k in e.list_keys('e5.oper.invariant/') if '128_172_lsm' in k]
    if not keys:
        sys.exit('land-sea mask file not found under e5.oper.invariant/')
    p = e.download(keys[0], os.path.join(out, 'lsm.nc'))
    with netCDF4.Dataset(p) as ds:
        lat, lon, ilat, ilon = e.window(ds)
        var = e.data_var(ds, 3)
        lsm = np.array(var[0, ilat[0]:ilat[-1] + 1, ilon[0]:ilon[-1] + 1], dtype=np.float32)
    os.remove(p)
    np.savez_compressed(os.path.join(out, 'era5_landmask.npz'), lat=lat[ilat], lon=lon[ilon], lsm=lsm)
    print(f'wrote era5_landmask.npz {lsm.shape}; land fraction (lsm>=0.5): {(lsm >= 0.5).mean():.2f}')
    for label, (la, lo) in {'Brisbane': (-27.47, 153.03), 'Alice Springs': (-23.7, 133.88), 'Tasman Sea': (-35.0, 156.0), 'Perth': (-31.95, 115.86)}.items():
        i = int(np.abs(lat[ilat] - la).argmin()); j = int(np.abs(lon[ilon] - lo).argmin())
        print(f'  {label}: lsm={lsm[i, j]:.2f}')

if __name__ == '__main__':
    main()
