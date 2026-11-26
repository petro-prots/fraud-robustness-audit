import json

import numpy as np
import pandas as pd
import pytest

from fraudaudit.data import period
from fraudaudit.features.base import BaseEncoder
from fraudaudit.features.columns import BASE_COLUMNS, CATEGORIES, FEATURE_COLUMNS


def _fit_df(sparkov_like):
    return sparkov_like[period(sparkov_like) == "fit"]


def _row(sparkov_like, **over):
    r = sparkov_like.iloc[[0]].copy()
    for k, v in over.items():
        r[k] = v
    return r


def test_27_base_38_feature_columns(sparkov_like):
    assert len(BASE_COLUMNS) == 27 and len(set(BASE_COLUMNS)) == 27
    assert BASE_COLUMNS[-2:] == ("hour_sin", "hour_cos")
    enc = BaseEncoder().fit(_fit_df(sparkov_like))
    out = enc.transform(sparkov_like)
    assert list(out.columns) == list(BASE_COLUMNS)
    assert (out.dtypes == "float32").all()
    assert out.index.equals(sparkov_like.index)
    assert not out.isna().any().any()
    assert len(FEATURE_COLUMNS) == 38


def test_no_calendar_features():
    # ruling R23: trans_year x trans_month put the test period outside the training support
    assert "trans_month" not in FEATURE_COLUMNS and "trans_year" not in FEATURE_COLUMNS
    assert {"age", "hour_sin", "hour_cos"} <= set(BASE_COLUMNS)


def test_age_rounding(sparkov_like):
    enc = BaseEncoder().fit(_fit_df(sparkov_like))
    a = enc.transform(_row(sparkov_like, dob="2000-06-15", ts=pd.Timestamp("2020-06-14 23:59:00")))
    assert a["age"].iloc[0] == 20
    # date part only: time of day is ignored; 20.0 y exactly -> 20
    b = enc.transform(_row(sparkov_like, dob="2000-06-15", ts=pd.Timestamp("2020-06-15 00:00:00")))
    assert b["age"].iloc[0] == 20
    assert "trans_month" not in b.columns and "trans_year" not in b.columns


@pytest.mark.parametrize("ts, want_sin, want_cos", [
    ("2020-06-14 00:00:00", 0.0, 1.0),
    ("2020-06-14 06:00:00", 1.0, 0.0),
    ("2020-06-14 12:00:00", 0.0, -1.0),
    # h = 23 + 59/60: angle = 2*pi - 2*pi/1440
    ("2020-06-14 23:59:00", -0.0043633093, 0.9999904807),
])
def test_hour_sin_cos(sparkov_like, ts, want_sin, want_cos):
    enc = BaseEncoder().fit(_fit_df(sparkov_like))
    out = enc.transform(_row(sparkov_like, ts=pd.Timestamp(ts)))
    assert out["hour_sin"].iloc[0] == pytest.approx(want_sin, abs=1e-7)
    assert out["hour_cos"].iloc[0] == pytest.approx(want_cos, abs=1e-7)


def test_hour_uses_seconds(sparkov_like):
    enc = BaseEncoder().fit(_fit_df(sparkov_like))
    out = enc.transform(_row(sparkov_like, ts=pd.Timestamp("2020-06-14 03:20:15")))
    h = 3 + 20 / 60 + 15 / 3600
    assert out["hour_sin"].iloc[0] == np.float32(np.sin(2 * np.pi * h / 24))
    assert out["hour_cos"].iloc[0] == np.float32(np.cos(2 * np.pi * h / 24))


def test_distance_rounding(sparkov_like):
    enc = BaseEncoder().fit(_fit_df(sparkov_like))
    r = _row(sparkov_like, lat=40.0, merch_lat=40.12345, long=-80.0, merch_long=-80.5)
    out = enc.transform(r)
    assert out["latitudinal_distance"].iloc[0] == pytest.approx(0.123, abs=1e-6)
    assert out["longitudinal_distance"].iloc[0] == pytest.approx(0.5, abs=1e-6)
    r = _row(sparkov_like, lat=40.0, merch_lat=39.87655)
    assert enc.transform(r)["latitudinal_distance"].iloc[0] == pytest.approx(0.123, abs=1e-6)


def test_onehot_drop_first(sparkov_like):
    enc = BaseEncoder().fit(_fit_df(sparkov_like))
    cats = list(CATEGORIES)
    first = enc.transform(_row(sparkov_like, category=cats[0], gender="F"))
    cat_cols = [c for c in BASE_COLUMNS if c.startswith("category_")]
    assert len(cat_cols) == 13
    assert (first[cat_cols + ["gender_M"]].to_numpy() == 0).all()
    second = enc.transform(_row(sparkov_like, category=cats[1], gender="M"))
    assert second[f"category_{cats[1]}"].iloc[0] == 1
    assert second[cat_cols].to_numpy().sum() == 1
    assert second["gender_M"].iloc[0] == 1


def test_codes_first_appearance_and_prefix(sparkov_like):
    fit = _fit_df(sparkov_like)
    enc = BaseEncoder().fit(fit)
    names = list(dict.fromkeys(fit["merchant"].str.removeprefix("fraud_")))
    for i in (0, len(names) - 1):
        out = enc.transform(_row(sparkov_like, merchant="fraud_" + names[i]))
        assert out["merchant"].iloc[0] == i
        out = enc.transform(_row(sparkov_like, merchant=names[i]))
        assert out["merchant"].iloc[0] == i


def test_single_row_equals_batch(sparkov_like):
    enc = BaseEncoder().fit(_fit_df(sparkov_like))
    batch = enc.transform(sparkov_like)
    for i in (0, 17, len(sparkov_like) - 1):
        one = enc.transform(sparkov_like.iloc[[i]])
        pd.testing.assert_frame_equal(one, batch.iloc[[i]])


def test_json_roundtrip_equal_transform(sparkov_like):
    enc = BaseEncoder().fit(_fit_df(sparkov_like))
    d = json.loads(json.dumps(enc.to_json()))
    enc2 = BaseEncoder.from_json(d)
    pd.testing.assert_frame_equal(enc.transform(sparkov_like), enc2.transform(sparkov_like))
    assert enc2.to_json() == enc.to_json()


def test_unknown_merchant_raises(sparkov_like):
    enc = BaseEncoder().fit(_fit_df(sparkov_like))
    with pytest.raises(ValueError, match="merchant"):
        enc.transform(_row(sparkov_like, merchant="fraud_Nowhere Ltd"))


def test_unknown_state_and_category_raise(sparkov_like):
    enc = BaseEncoder().fit(_fit_df(sparkov_like))
    with pytest.raises(ValueError, match="state"):
        enc.transform(_row(sparkov_like, state="ZZ"))
    with pytest.raises(ValueError, match="category"):
        enc.transform(_row(sparkov_like, category="nope"))


def test_transform_before_fit_raises(sparkov_like):
    with pytest.raises(ValueError):
        BaseEncoder().transform(sparkov_like)
