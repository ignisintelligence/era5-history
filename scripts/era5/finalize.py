#!/usr/bin/env python3
"""
Stage 2: yearly ERA5 files (era5_daily.py) -> the two datasets the app reads.

  python finalize.py --in <folder with era5_daily_2000..2020.npz and era5_landmask.npz> --out <folder>

1. Drought factor per day and point: the same KBDI calculation as buildPrepareRaster (daily max temp + rain).
2. FFDI of each day = FFDI of its worst hour (temperature / humidity / wind at that hour, that day's drought factor).
3. Prepare: for every 0.15 degree land cell, the 20 highest-FFDI days of the nearest ERA5 point, with average / max / rating.
4. Ignite: one history file per land point in the format enterpriseComputeBurnAreaStats already reads
   (startDate, tmax, rhMin, rhMax, windMax, rain), where tmax / rhMin / windMax are the values AT THE WORST HOUR.
Outputs go under  <out>/files/...  (to be uploaded to the app's files bucket) plus the record lists to import.
"""
import argparse, glob, gzip, json, os, datetime as dt
import numpy as np

SCALES = {'tmax': 100, 'rh_min': 100, 'rh_max': 100, 'wind_max': 100, 'rain': 10, 'w_temp': 100, 'w_rh': 100, 'w_wind': 100, 'w_dir': 10, 'w_hour': 1}
CELL = 0.15
PREPARE_LAND, HISTORY_LAND = 0.5, 0.2   # land fraction needed for a Prepare cell / an Ignite history point
SANE = {'tmax': (-30, 55), 'rh_min': (0, 100), 'rh_max': (0, 100), 'wind_max': (0, 250), 'rain': (0, 800),
        'w_temp': (-30, 55), 'w_rh': (0, 100), 'w_wind': (0, 250), 'w_dir': (0, 360), 'w_hour': (0, 23)}


def unpack(z, key):
    a = z[key].astype(np.float32)
    bad = a == -32768
    a = a / SCALES[key]
    a[bad] = np.nan
    return a


def load_years(folder):
    files = sorted(glob.glob(os.path.join(folder, '**', 'era5_daily_*.npz'), recursive=True))
    if not files:
        raise SystemExit('no era5_daily_*.npz files found')
    years = [int(os.path.basename(f)[11:15]) for f in files]
    if years != list(range(years[0], years[0] + len(years))):
        raise SystemExit(f'years are not consecutive: {years}')
    lat = lon = None
    parts, dates = {}, []
    for f in files:
        z = np.load(f, allow_pickle=False)           # never allow pickles: data only
        if lat is None:
            lat, lon = z['lat'], z['lon']
        elif not (np.array_equal(lat, z['lat']) and np.array_equal(lon, z['lon'])):
            raise SystemExit(f'{f}: grid differs from the other years')
        dates += [str(d) for d in z['dates']]
        for k in SCALES:
            a = unpack(z, k)
            lo, hi = SANE[k]
            if np.nanmin(a) < lo - 1e-3 or np.nanmax(a) > hi + 1e-3:
                raise SystemExit(f'{f}: {k} outside real-world range ({np.nanmin(a)}..{np.nanmax(a)})')
            parts.setdefault(k, []).append(a)
    return lat, lon, dates, {k: np.concatenate(v, axis=0) for k, v in parts.items()}, years


def drought_factor(tmax, rain):
    """Same simplified KBDI -> McArthur DF as buildPrepareRaster.computeDailyDF, for every point at once."""
    n = tmax.shape[0]
    kbdi = np.zeros(tmax.shape[1:], np.float32)
    out = np.empty_like(tmax)
    for i in range(n):
        t = np.nan_to_num(tmax[i], nan=0.0); r = np.nan_to_num(rain[i], nan=0.0)
        kbdi = np.maximum(0.0, kbdi - np.maximum(0.0, r - 5.0))
        kbdi = np.where(t > 15.5, np.minimum(203.0, kbdi + (203.0 - kbdi) * 0.003 * (t - 15.5)), kbdi)
        df = 10.5 * (1 - np.exp(-(kbdi + 30) / 40))
        out[i] = np.round(np.clip(df, 0.1, 10.0), 1)
    return out


def ffdi(temp, rh, wind, df):
    f = 2 * np.exp(0.987 * np.log(np.maximum(1, df)) - 0.0345 * np.maximum(1, rh) + 0.0338 * temp + 0.0234 * wind - 0.45)
    return np.round(f * 10) / 10


def rating(v):
    return 'Low-Moderate' if v < 12 else 'High' if v < 32 else 'Very High' if v < 50 else 'Severe' if v < 75 else 'Extreme' if v < 100 else 'Catastrophic'


def top20(days_ffdi, dates, d, i, j):
    order = np.argsort(-days_ffdi[:, i, j], kind='stable')[:20]
    out = []
    for k in order:
        if not np.isfinite(days_ffdi[k, i, j]):
            continue
        out.append({'date': dates[k], 'ffdi': round(float(days_ffdi[k, i, j]), 1), 'temp': round(float(d['w_temp'][k, i, j]), 1),
                    'rh': int(round(float(d['w_rh'][k, i, j]))), 'wind': int(round(float(d['w_wind'][k, i, j]))),
                    'windDir': int(round(float(d['w_dir'][k, i, j]))), 'rain': round(float(d['rain'][k, i, j]), 1),
                    'hour': int(d['w_hour'][k, i, j])})
    return out


def write_gz(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=9) as f:
        json.dump(obj, f, separators=(',', ':'))


def r1(a):
    return [None if not np.isfinite(x) else round(float(x), 1) for x in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lat, lon, dates, d, years = load_years(a.inp)
    lm = np.load(glob.glob(os.path.join(a.inp, '**', 'era5_landmask.npz'), recursive=True)[0], allow_pickle=False)
    if not (np.array_equal(lm['lat'], lat) and np.array_equal(lm['lon'], lon)):
        raise SystemExit('land mask grid differs from the daily files')
    lsm = lm['lsm']
    print(f'{len(dates)} days ({years[0]}-{years[-1]}), grid {len(lat)} x {len(lon)}')

    df = drought_factor(d['tmax'], d['rain'])
    f = ffdi(d['w_temp'], d['w_rh'], d['w_wind'], df)
    print('drought factor and FFDI done; FFDI max', float(np.nanmax(f)))

    base = os.path.join(a.out, 'files')
    # ── Ignite history files (every point with some land) ──
    hist_records, hist_idx = [], 0
    for i in range(len(lat)):
        for j in range(len(lon)):
            if lsm[i, j] < HISTORY_LAND:
                continue
            key = f'{lat[i]:.2f}_{lon[j]:.2f}'
            raw = {'startDate': dates[0], 'source': 'ERA5 worst-hour', 'tmax': r1(d['w_temp'][:, i, j]), 'rhMin': r1(d['w_rh'][:, i, j]),
                   'rhMax': r1(d['rh_max'][:, i, j]), 'windMax': r1(d['w_wind'][:, i, j]), 'rain': r1(d['rain'][:, i, j])}
            write_gz(os.path.join(base, 'history', 'era5', f'h_{key}.json.gz'), raw)
            hist_records.append({'geoKey': f'{lat[i]:.2f},{lon[j]:.2f}', 'lat': float(lat[i]), 'lng': float(lon[j]), 'status': 'complete',
                                 'file_key': f'files/history/era5/h_{key}.json.gz'})
    print(f'{len(hist_records)} Ignite history files')

    # ── Prepare: top-20 detail per land point, cells on the 0.15 degree grid ──
    detail, summary = {}, {}
    for i in range(len(lat)):
        for j in range(len(lon)):
            if lsm[i, j] < PREPARE_LAND:
                continue
            t = top20(f, dates, d, i, j)
            if not t:
                continue
            avg = round(sum(x['ffdi'] for x in t) / len(t), 1)
            detail[(i, j)] = f'files/prepare/era5/p_{lat[i]:.2f}_{lon[j]:.2f}.json.gz'
            summary[(i, j)] = (avg, t[0]['ffdi'])
            write_gz(os.path.join(base, 'prepare', 'era5', f'p_{lat[i]:.2f}_{lon[j]:.2f}.json.gz'), t)
    lat_s, lon_s = np.sort(lat), lon
    cells = []
    la = np.ceil(lat.min() / CELL) * CELL
    while la <= lat.max():
        lo = np.ceil(lon.min() / CELL) * CELL
        while lo <= lon.max():
            i = int(np.abs(lat - la).argmin()); j = int(np.abs(lon - lo).argmin())
            if (i, j) in summary:
                avg, mx = summary[(i, j)]
                cells.append({'geoKey': f'{la:.2f},{lo:.2f}', 'lat': round(float(la), 2), 'lng': round(float(lo), 2), 'avgFfdi': avg,
                              'maxFfdi': mx, 'rating': rating(avg), 'file_key': detail[(i, j)], 'status': 'complete'})
            lo += CELL
        la += CELL
    print(f'{len(cells)} Prepare cells from {len(summary)} land points')
    os.makedirs(a.out, exist_ok=True)
    json.dump(cells, open(os.path.join(a.out, 'PrepareGridCell.records.json'), 'w'))
    json.dump(hist_records, open(os.path.join(a.out, 'HistoryGrid.records.json'), 'w'))
    ratings = {}
    for c in cells:
        ratings[c['rating']] = ratings.get(c['rating'], 0) + 1
    print('Prepare rating counts:', ratings)


if __name__ == '__main__':
    main()
