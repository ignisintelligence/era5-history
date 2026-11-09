import numpy as np, os, json, gzip, finalize as fz

def test_ffdi_and_rating():
    assert fz.ffdi(np.float32(30), np.float32(30), np.float32(20), np.float32(8)) == 15.5
    assert fz.rating(11.9) == 'Low-Moderate' and fz.rating(32) == 'Very High' and fz.rating(120) == 'Catastrophic'

def test_df_rises_in_dry_heat_and_falls_with_rain():
    n = 60
    t = np.full((n, 1, 1), 38, np.float32); r = np.zeros((n, 1, 1), np.float32)
    df = fz.drought_factor(t, r)
    assert df[-1, 0, 0] > df[0, 0, 0] and 0.1 <= df.min() and df.max() <= 10
    r[30] = 150
    df2 = fz.drought_factor(t, r)
    assert df2[31, 0, 0] < df2[29, 0, 0]

def test_pipeline_end_to_end(tmp_path):
    lat = np.array([-27.5, -27.25], np.float32); lon = np.array([153.0, 153.25], np.float32)
    inp = tmp_path / 'in'; inp.mkdir()
    for y in (2000, 2001):
        n = 366 if y == 2000 else 365
        rng = np.random.RandomState(y)
        shape = (n, 2, 2)
        vals = {'tmax': rng.uniform(15, 40, shape), 'rh_min': rng.uniform(10, 80, shape), 'rh_max': rng.uniform(60, 100, shape),
                'wind_max': rng.uniform(5, 50, shape), 'rain': rng.uniform(0, 3, shape), 'w_temp': rng.uniform(15, 40, shape),
                'w_rh': rng.uniform(10, 80, shape), 'w_wind': rng.uniform(5, 50, shape), 'w_dir': rng.uniform(0, 360, shape), 'w_hour': rng.randint(0, 24, shape)}
        packed = {k: np.round(v * fz.SCALES[k]).astype(np.int16) for k, v in vals.items()}
        import datetime as dt
        dates = np.array([(dt.date(y, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(n)])
        np.savez_compressed(inp / f'era5_daily_{y}.npz', lat=lat, lon=lon, dates=dates, **packed)
    np.savez_compressed(inp / 'era5_landmask.npz', lat=lat, lon=lon, lsm=np.array([[1.0, 0.0], [0.6, 0.3]], np.float32))
    import sys
    sys.argv = ['finalize', '--in', str(inp), '--out', str(tmp_path / 'out'), '--strip', '1']
    fz.main()
    cells = json.load(open(tmp_path / 'out' / 'PrepareGridCell.records.json'))
    hist = json.load(open(tmp_path / 'out' / 'HistoryGrid.records.json'))
    assert len(hist) == 3 and cells and all(c['status'] == 'complete' for c in cells)
    f = tmp_path / 'out' / hist[0]['file_key']
    raw = json.load(gzip.open(f, 'rt'))
    assert len(raw['tmax']) == 731 and raw['startDate'] == '2000-01-01' and set(raw) >= {'tmax', 'rhMin', 'rhMax', 'windMax', 'rain'}
    top = json.load(gzip.open(tmp_path / 'out' / cells[0]['file_key'], 'rt'))
    assert len(top) == 20 and top[0]['ffdi'] >= top[-1]['ffdi'] and cells[0]['maxFfdi'] == top[0]['ffdi']
