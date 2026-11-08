import numpy as np, era5_daily as e

def test_rh_and_wind():
    assert abs(float(e.rh_from_dewpoint(np.float32(20), np.float32(20))) - 100) < 0.01
    assert 45 < float(e.rh_from_dewpoint(np.float32(30), np.float32(18))) < 50
    s, d = e.wind_from_uv(np.float32(0), np.float32(-10))   # wind blowing towards the south = from the north
    assert abs(float(s) - 36) < 1e-3 and abs(float(d) - 0) < 1e-3 or abs(float(d) - 360) < 1e-3
    s, d = e.wind_from_uv(np.float32(-10), np.float32(0))   # blowing towards the west = from the east
    assert abs(float(d) - 90) < 1e-3

def test_worst_hour_is_coincident_not_separate_maxima():
    # hour 5: hot but humid & calm; hour 15: hot-ish, dry, windy; separate extremes would mix them
    T = np.full((24, 1, 1), 20, np.float32); RH = np.full((24, 1, 1), 60, np.float32)
    W = np.full((24, 1, 1), 5, np.float32); D = np.full((24, 1, 1), 180, np.float32); R = np.zeros((24, 1, 1), np.float32)
    T[5] = 35; RH[5] = 70; W[5] = 5
    T[15] = 30; RH[15] = 20; W[15] = 40; D[15] = 270
    r = e.aggregate_days(T, RH, W, D, R)
    assert int(r['w_hour'][0, 0, 0]) == 15
    assert r['w_temp'][0, 0, 0] == 30 and r['w_rh'][0, 0, 0] == 20 and r['w_wind'][0, 0, 0] == 40 and r['w_dir'][0, 0, 0] == 270
    assert r['tmax'][0, 0, 0] == 35 and r['rh_min'][0, 0, 0] == 20 and r['wind_max'][0, 0, 0] == 40  # the separate maxima differ

def test_multi_day_and_rain():
    n = 48
    T = np.random.RandomState(1).uniform(10, 40, (n, 2, 3)).astype(np.float32)
    RH = np.full_like(T, 30); W = np.full_like(T, 10); D = np.zeros_like(T); R = np.ones_like(T)
    r = e.aggregate_days(T, RH, W, D, R)
    assert r['rain'].shape == (2, 2, 3) and float(r['rain'][0, 0, 0]) == 24
    assert np.allclose(r['tmax'][1], T[24:].max(axis=0))

def test_utc_offset_bands():
    lon = np.array([115.9, 133.9, 138.6, 153.0, 147.3])
    assert list(e.utc_offset_for_lon(lon)) == [8, 9, 9, 10, 10]
