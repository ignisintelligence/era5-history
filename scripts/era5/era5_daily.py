#!/usr/bin/env python3
"""
ERA5 -> daily "worst coincident hour" history for Australia.

Reads hourly ERA5 from the public NSF NCAR bucket (s3://nsf-ncar-era5, anonymous, no login), keeps only the
Australian window, and for every local standard-time day and every 0.25 degree point works out:
  - the worst hour of the day, i.e. the hour with the highest FFDI (the drought factor is constant within a
    day, so the worst hour is the hour with the highest  -0.0345*max(1,RH) + 0.0338*T + 0.0234*W ),
  - temperature / humidity / wind speed / wind direction in that hour,
  - the day's max temperature, min and max humidity, max wind and total rain (for the other Ignite checks).
The drought factor and the final FFDI are applied in the next stage (they need the whole 2000-2020 series).

Variables (hourly, 0.25 deg):  an.sfc 128_167_2t, 128_168_2d, 128_165_10u, 128_166_10v
Rain: fc.sfc.accumu 128_142_lsp + 128_143_cp (total precipitation = large-scale + convective, accumulated
since forecast start, metres), de-accumulated to hourly.
"""
import argparse, calendar, datetime as dt, os, subprocess, sys, urllib.request, xml.etree.ElementTree as ET
import numpy as np

BASE = 'https://nsf-ncar-era5.s3.amazonaws.com'
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = -44.5, -9.0, 112.0, 154.5   # all of Australia incl. Tasmania and Cape York
BASE_OFFSET = 10   # eastern standard time; raw hourly arrays start at local midnight for UTC+10


def utc_offset_for_lon(lon):
    """Local STANDARD time offset (hours) used to cut days: WA +8, SA/NT +9, east coast +10. No daylight saving,
    so a 'day' is the same 24 hours all year."""
    return np.where(lon < 127.5, 8, np.where(lon < 142.5, 9, 10))

SCALES = {  # stored as int16 = value * scale
    'tmax': 100, 'rh_min': 100, 'rh_max': 100, 'wind_max': 100, 'rain': 10,
    'w_temp': 100, 'w_rh': 100, 'w_wind': 100, 'w_dir': 10, 'w_hour': 1,
}


# ── pure computation (unit-testable, no I/O) ─────────────────────────────────
def rh_from_dewpoint(t_c, td_c):
    """Relative humidity (%) from air and dewpoint temperature in C (Alduchov-Eskridge Magnus form)."""
    es = lambda x: 6.1094 * np.exp(17.625 * x / (x + 243.04))
    return np.clip(100.0 * es(td_c) / es(t_c), 0.0, 100.0)


def wind_from_uv(u, v):
    """Speed in km/h and meteorological direction (degrees the wind blows FROM)."""
    speed = np.sqrt(u * u + v * v) * 3.6
    direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return speed, direction


def aggregate_days(t_c, rh, wind, wdir, rain):
    """Inputs shaped (days*24, lat, lon), hour 0 = local midnight. Returns dict of (days, lat, lon) arrays."""
    n, ny, nx = t_c.shape
    assert n % 24 == 0, 'hours must be a whole number of days'
    d = n // 24
    sh = (d, 24, ny, nx)
    T, RH, W, D, R = (a.reshape(sh) for a in (t_c, rh, wind, wdir, rain))
    s = -0.0345 * np.maximum(1.0, RH) + 0.0338 * T + 0.0234 * W
    s = np.where(np.isfinite(s), s, -np.inf)
    idx = np.argmax(s, axis=1)[:, None]
    at = lambda a: np.take_along_axis(a, idx, axis=1)[:, 0]
    return {
        'tmax': T.max(axis=1), 'rh_min': RH.min(axis=1), 'rh_max': RH.max(axis=1), 'wind_max': W.max(axis=1),
        'rain': R.sum(axis=1),
        'w_temp': at(T), 'w_rh': at(RH), 'w_wind': at(W), 'w_dir': at(D), 'w_hour': idx[:, 0].astype(np.float32),
    }


# ── data access ──────────────────────────────────────────────────────────────
def download(key, dest):
    url = f'{BASE}/{key}'
    print(f'  downloading {key}', flush=True)
    subprocess.run(['curl', '-fsSL', '--retry', '8', '--retry-delay', '10', '--retry-all-errors', '-C', '-', '-o', dest, url], check=True)
    return dest


def list_keys(prefix):
    keys, token = [], None
    while True:
        url = f'{BASE}/?list-type=2&prefix={prefix}' + (f'&continuation-token={token}' if token else '')
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        ns = {'s': root.tag.split('}')[0].strip('{')}
        keys += [k.text for k in root.findall('.//s:Contents/s:Key', ns)]
        if root.findtext('s:IsTruncated', default='false', namespaces=ns) == 'true':
            token = root.findtext('s:NextContinuationToken', namespaces=ns)
        else:
            return keys


def month_key_an(code, y, m):
    last = calendar.monthrange(y, m)[1]
    ym = f'{y}{m:02d}'
    return f'e5.oper.an.sfc/{ym}/e5.oper.an.sfc.{code}.ll025sc.{ym}0100_{ym}{last}23.nc'


def window(ds):
    lat = np.array(ds.variables['latitude'][:]); lon = np.array(ds.variables['longitude'][:])
    ilat = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    ilon = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
    return lat, lon, ilat, ilon


def data_var(ds, ndim):
    for name, v in ds.variables.items():
        if v.ndim == ndim and 'latitude' in v.dimensions and 'longitude' in v.dimensions:
            return v
    raise RuntimeError(f'no {ndim}-d lat/lon variable in {list(ds.variables)}')


def to_dt(var):
    import netCDF4
    vals = netCDF4.num2date(var[:], var.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    return np.array([np.datetime64(v.replace(tzinfo=None), 'h') for v in np.atleast_1d(vals)])


def read_hourly(path, t0, nh, out, grid):
    """an.sfc month file -> fills out[(valid_hour - t0)] for hours inside the wanted range."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        lat, lon, ilat, ilon = window(ds)
        grid.setdefault('lat', lat[ilat]); grid.setdefault('lon', lon[ilon])
        var = data_var(ds, 3)
        times = to_dt(ds.variables['time'])
        off = ((times - np.datetime64(t0, 'h')) / np.timedelta64(1, 'h')).astype(int)
        sel = np.where((off >= 0) & (off < nh))[0]
        if sel.size == 0:
            return
        block = np.array(var[sel[0]:sel[-1] + 1, ilat[0]:ilat[-1] + 1, ilon[0]:ilon[-1] + 1], dtype=np.float32)
        out[off[sel[0]:sel[-1] + 1]] = block


def read_accum(path, t0, nh, out, grid):
    """accumu half-month file (lsp or cp) -> adds hourly rain in mm into out."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        lat, lon, ilat, ilon = window(ds)
        var = data_var(ds, 4)
        init = to_dt(ds.variables['forecast_initial_time'])
        steps = np.array(ds.variables['forecast_hour'][:]).astype(int)
        acc = np.array(var[:, :, ilat[0]:ilat[-1] + 1, ilon[0]:ilon[-1] + 1], dtype=np.float32)
        hourly = np.diff(acc, axis=1, prepend=0.0)
        hourly[:, 0] = acc[:, 0]
        hourly = np.clip(hourly, 0.0, None) * 1000.0  # m -> mm
        for i, t in enumerate(init):
            for k, h in enumerate(steps):
                o = int((t - np.datetime64(t0, 'h')) / np.timedelta64(1, 'h')) + int(h)
                if 0 <= o < nh:
                    out[o] += hourly[i, k]


def build(t0, ndays, tmp):
    """t0: UTC datetime of local (UTC+10) midnight of the first day. Raw hourly data is kept on disk (memmap)
    so a whole year fits in a small runner; days are cut per longitude band using that band's standard time."""
    nh = ndays * 24 + 2          # +2: western bands start later in UTC
    t1 = t0 + dt.timedelta(hours=nh)
    months, y, m = [], t0.year, t0.month
    while (y, m) <= (t1.year, t1.month):
        months.append((y, m)); m += 1
        if m == 13: y, m = y + 1, 1
    # accumu forecasts start at 06/18 UTC, so the previous month's folder holds the first hours
    acc_months = [(t0.year - (t0.month == 1), 12 if t0.month == 1 else t0.month - 1)] + months
    grid = {}
    arrays = {}
    for code in ('128_167_2t', '128_168_2d', '128_165_10u', '128_166_10v'):
        for (yy, mm) in months:
            p = download(month_key_an(code, yy, mm), os.path.join(tmp, 'f.nc'))
            if code not in arrays:
                with __import__('netCDF4').Dataset(p) as ds:
                    _, _, ilat, ilon = window(ds)
                arrays[code] = np.lib.format.open_memmap(os.path.join(tmp, code + '.npy'), mode='w+', dtype=np.float32, shape=(nh, len(ilat), len(ilon)))
                arrays[code][:] = np.nan
            read_hourly(p, t0, nh, arrays[code], grid)
            os.remove(p)
    rain = np.lib.format.open_memmap(os.path.join(tmp, 'rain.npy'), mode='w+', dtype=np.float32, shape=arrays['128_167_2t'].shape)
    rain[:] = 0
    for code in ('128_142_lsp', '128_143_cp'):
        for (yy, mm) in acc_months:
            for key in list_keys(f'e5.oper.fc.sfc.accumu/{yy}{mm:02d}/e5.oper.fc.sfc.accumu.{code}.'):
                p = download(key, os.path.join(tmp, 'f.nc'))
                read_accum(p, t0, nh, rain, grid)
                os.remove(p)
    lon = grid['lon']
    offs = utc_offset_for_lon(lon)
    res = {}
    for o in (8, 9, 10):
        cols = np.where(offs == o)[0]
        if cols.size == 0:
            continue
        lo, hi = cols[0], cols[-1] + 1
        start = BASE_OFFSET - o     # hours after t0 at which this band's local midnight falls
        parts = {}
        BLK = 30                    # days per block keeps memory small
        for d0 in range(0, ndays, BLK):
            d1 = min(ndays, d0 + BLK)
            sl = slice(start + d0 * 24, start + d1 * 24)
            t_c = arrays['128_167_2t'][sl, :, lo:hi] - 273.15
            td_c = arrays['128_168_2d'][sl, :, lo:hi] - 273.15
            if np.isnan(t_c).any() or np.isnan(td_c).any():
                raise RuntimeError(f'missing temperature data in band UTC+{o} days {d0}-{d1}')
            rh = rh_from_dewpoint(t_c, td_c)
            wind, wdir = wind_from_uv(arrays['128_165_10u'][sl, :, lo:hi], arrays['128_166_10v'][sl, :, lo:hi])
            r = aggregate_days(t_c, rh, wind, wdir, np.asarray(rain[sl, :, lo:hi]))
            for k, v in r.items():
                parts.setdefault(k, []).append(v)
        for k, v in parts.items():
            res.setdefault(k, []).append(np.concatenate(v, axis=0))
    res = {k: np.concatenate(v, axis=2) for k, v in res.items()}
    return grid, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int)
    ap.add_argument('--smoke', action='store_true', help='one week of January 2000 only, prints a sanity report')
    ap.add_argument('--out', default='out')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, 'tmp'); os.makedirs(tmp, exist_ok=True)
    if a.smoke:
        first = dt.date(2000, 1, 2); ndays = 7
    else:
        first = dt.date(a.year, 1, 1); ndays = 366 if calendar.isleap(a.year) else 365
    t0 = dt.datetime(first.year, first.month, first.day) - dt.timedelta(hours=BASE_OFFSET)
    grid, res = build(t0, ndays, tmp)
    dates = np.array([(first + dt.timedelta(days=i)).isoformat() for i in range(ndays)])
    packed = {k: np.round(np.nan_to_num(v, nan=-32768 / SCALES[k]) * SCALES[k]).astype(np.int16) for k, v in res.items()}
    name = 'era5_smoke.npz' if a.smoke else f'era5_daily_{first.year}.npz'
    np.savez_compressed(os.path.join(a.out, name), lat=grid['lat'], lon=grid['lon'], dates=dates, **packed)
    print(f'wrote {name}: {ndays} days, {len(grid["lat"])} x {len(grid["lon"])} points')
    if a.smoke:
        lat, lon = grid['lat'], grid['lon']
        for label, (la, lo) in {'Brisbane': (-27.47, 153.03), 'Longreach': (-23.44, 144.25), 'Perth': (-31.95, 115.86), 'Alice Springs': (-23.70, 133.88), 'Sydney': (-33.87, 151.21), 'Hobart': (-42.88, 147.33)}.items():
            i = int(np.abs(lat - la).argmin()); j = int(np.abs(lon - lo).argmin())
            print(f'\n{label} (grid {lat[i]:.2f},{lon[j]:.2f}):')
            for d in range(ndays):
                print('  ', dates[d], ' '.join(f'{k}={float(res[k][d, i, j]):.1f}' for k in res))
        print('\nrange check:')
        for k, v in res.items():
            print(f'  {k:9s} min {np.nanmin(v):8.2f}  max {np.nanmax(v):8.2f}  mean {np.nanmean(v):8.2f}')


if __name__ == '__main__':
    main()
