"""Fixed feature order of the model: 27 base features (the 27 of the bachelor model without the
calendar features ``trans_month``/``trans_year``, + time of day ``hour_sin``/``hour_cos``) and
7 windowed aggregates.

The calendar features were dropped (ruling R23): ``trans_year`` x ``trans_month`` put the test
period (Jun-Dec 2020) outside the training support, so models extrapolated on test."""
from __future__ import annotations

# The 14 Sparkov categories, sorted; the first one is the baseline dropped by
# pd.get_dummies(drop_first=True). Fixed here, so the column order never depends on the fit data.
CATEGORIES = (
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
    "shopping_net", "shopping_pos", "travel",
)
CATEGORY_DUMMIES = tuple(f"category_{c}" for c in CATEGORIES[1:])

BASE_COLUMNS: tuple[str, ...] = (
    "merchant", "amt", "state", "lat", "long", "city_pop", "merch_lat", "merch_long",
    "age", "latitudinal_distance", "longitudinal_distance",
    *CATEGORY_DUMMIES,
    "gender_M",
    "hour_sin", "hour_cos",  # time of day on the unit circle (not in the bachelor set; ruling R16)
)

AGG_COLUMNS: tuple[str, ...] = (
    "cnt_1d", "cnt_7d", "cnt_30d", "mean_amt_1d", "mean_amt_7d", "mean_amt_30d", "secs_since_prev",
)

FEATURE_COLUMNS: tuple[str, ...] = BASE_COLUMNS + AGG_COLUMNS
