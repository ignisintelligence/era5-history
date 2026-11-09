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


def load_years(folder, work):
    """Reads each yearly file once and appends it to on-disk arrays (days, lat, lon), so a strip of the country
    can later be processed without holding the whole 21-year series in memory."""
    files = sorted(glob.glob(os.path.join(folder, '**', 'era5_daily_*.npz'), recursive=True))
    if not files:
        raise SystemExit('no era5_daily_*.npz files found')
    years = [int(os.path.basename(f)[11:15]) for f in files]
    if years != list(range(years[0], years[0] + len(years))):
        raise SystemExit(f'years are not consecutive: {years}')
    os.makedirs(work, exist_ok=True)
    meta = []
    for f in files:                       # first pass: sizes only
        z = np.load(f, allow_pickle=False)
        meta.append((f, len(z['dates'])))
    ndays = sum(n for _, n in meta)
    lat = lon = None
    mm, dates, pos = {}, [], 0
    for f, n in meta:
        z = np.load(f, allow_pickle=False)           # never allow pickles: data only
        if lat is None:
            lat, lon = z['lat'], z['lon']
            for k in SCALES:
                mm[k] = np.lib.format.open_memmap(os.path.join(work, k + '.npy'), mode='w+', dtype=np.float32, shape=(ndays, len(lat), len(lon)))
        elif not (np.array_equal(lat, z['lat']) and np.array_equal(lon, z['lon'])):
            raise SystemExit(f'{f}: grid differs from the other years')
        dates += [str(x) for x in z['dates']]
        for k in SCALES:
            arr = unpack(z, k)
            lo, hi = SANE[k]
            if np.nanmin(arr) < lo - 1e-3 or np.nanmax(arr) > hi + 1e-3:
                raise SystemExit(f'{f}: {k} outside real-world range ({np.nanmin(arr)}..{np.nanmax(arr)})')
            mm[k][pos:pos + n] = arr
        pos += n
    for v in mm.values():
        v.flush()
    return lat, lon, dates, mm, years


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
    ap.add_argument('--strip', type=int, default=8, help='latitude rows processed at once')
    a = ap.parse_args()
    work = os.path.join(a.out, '_work')
    lat, lon, dates, mm, years = load_years(a.inp, work)
    lm = np.load(glob.glob(os.path.join(a.inp, '**', 'era5_landmask.npz'), recursive=True)[0], allow_pickle=False)
    if not (np.array_equal(lm['lat'], lat) and np.array_equal(lm['lon'], lon)):
        raise SystemExit('land mask grid differs from the daily files')
    lsm = lm['lsm']
    print(f'{len(dates)} days ({years[0]}-{years[-1]}), grid {len(lat)} x {len(lon)}', flush=True)

    base = os.path.join(a.out, 'files')
    hist_records, detail, summary = [], {}, {}
    top_ffdi = 0.0
    for i0 in range(0, len(lat), a.strip):
        i1 = min(len(lat), i0 + a.strip)
        if not ((lsm[i0:i1] >= min(HISTORY_LAND, PREPARE_LAND)).any()):
            continue
        d = {k: np.asarray(mm[k][:, i0:i1, :]) for k in SCALES}
        df = drought_factor(d['tmax'], d['rain'])
        f = ffdi(d['w_temp'], d['w_rh'], d['w_wind'], df)
        top_ffdi = max(top_ffdi, float(np.nanmax(f)))
        for ii in range(i1 - i0):
            gi = i0 + ii
            for j in range(len(lon)):
                land = lsm[gi, j]
                if land >= HISTORY_LAND:
                    key = f'{lat[gi]:.2f}_{lon[j]:.2f}'
                    raw = {'startDate': dates[0], 'source': 'ERA5 worst-hour', 'tmax': r1(d['w_temp'][:, ii, j]), 'rhMin': r1(d['w_rh'][:, ii, j]),
                           'rhMax': r1(d['rh_max'][:, ii, j]), 'windMax': r1(d['w_wind'][:, ii, j]), 'rain': r1(d['rain'][:, ii, j]),
                           'df': r1(df[:, ii, j]), 'ffdi': r1(f[:, ii, j]),
                           'windDir': r1(d['w_dir'][:, ii, j]), 'hour': [None if not np.isfinite(x) else int(x) for x in d['w_hour'][:, ii, j]]}
                    write_gz(os.path.join(base, 'history', 'era5', f'h_{key}.json.gz'), raw)
                    hist_records.append({'geoKey': f'{lat[gi]:.2f},{lon[j]:.2f}', 'lat': float(lat[gi]), 'lng': float(lon[j]), 'status': 'complete',
                                         'file_key': f'files/history/era5/h_{key}.json.gz'})
                if land >= PREPARE_LAND:
                    t = top20(f, dates, d, ii, j)
                    if t:
                        avg = round(sum(x['ffdi'] for x in t) / len(t), 1)
                        detail[(gi, j)] = f'files/prepare/era5/p_{lat[gi]:.2f}_{lon[j]:.2f}.json.gz'
                        summary[(gi, j)] = (avg, t[0]['ffdi'])
                        write_gz(os.path.join(base, 'prepare', 'era5', f'p_{lat[gi]:.2f}_{lon[j]:.2f}.json.gz'), t)
        print(f'  rows {i0}-{i1} of {len(lat)} done', flush=True)
    print(f'{len(hist_records)} Ignite history files; highest FFDI {top_ffdi}')

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
    json.dump(cells, open(os.path.join(a.out, 'PrepareGridCell.records.json'), 'w'))
    json.dump(hist_records, open(os.path.join(a.out, 'HistoryGrid.records.json'), 'w'))
    ratings = {}
    for c in cells:
        ratings[c['rating']] = ratings.get(c['rating'], 0) + 1
    print('Prepare rating counts:', ratings)
    import shutil
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
