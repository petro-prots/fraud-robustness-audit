"""Fixed feature order of the model (27 bachelor base features + 7 windowed aggregates)."""
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
    "age", "trans_month", "trans_year", "latitudinal_distance", "longitudinal_distance",
    *CATEGORY_DUMMIES,
    "gender_M",
)

AGG_COLUMNS: tuple[str, ...] = (
    "cnt_1d", "cnt_7d", "cnt_30d", "mean_amt_1d", "mean_amt_7d", "mean_amt_30d", "secs_since_prev",
)

FEATURE_COLUMNS: tuple[str, ...] = BASE_COLUMNS + AGG_COLUMNS
