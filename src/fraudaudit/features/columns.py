"""Fixed feature order of the model: 27 base features (the 27 of the bachelor model without the
calendar features ``trans_month``/``trans_year``, + time of day ``hour_sin``/``hour_cos``) and
11 windowed aggregates (the 7 of the plan + 4 pre-registered behavioural features of Task 23:
``cnt_1h``, ``sum_1h``, ``cnt_cat_24h``, ``amt_ratio_30d``) - 38 features.

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
    # behavioural features (Task 23): short window, category velocity, amount deviation
    "cnt_1h", "sum_1h", "cnt_cat_24h", "amt_ratio_30d",
)

FEATURE_COLUMNS: tuple[str, ...] = BASE_COLUMNS + AGG_COLUMNS

# Registered feature sets (name -> ordered columns). Every set is an ordered subset of FEATURE_COLUMNS: features
# are always computed as FEATURE_COLUMNS (batch and serving alike) and a model trained on a smaller set selects its
# columns by index (``FraudModel``). A model whose ``feature_columns`` are not exactly one of these is refused.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "bachelor+hour-calendar": BASE_COLUMNS + AGG_COLUMNS[:7],  # 34: the M2 baseline (models-no-behav, Task 20)
    "bachelor+hour-calendar+behav": FEATURE_COLUMNS,           # 38: the current model (Task 23)
}


def feature_set_name(columns) -> str:
    """Name of the registered feature set whose columns equal ``columns`` exactly (order included); ValueError
    otherwise (unknown, reordered, duplicated or missing columns)."""
    cols = tuple(columns)
    for name, fs in FEATURE_SETS.items():
        if cols == fs:
            return name
    raise ValueError(f"feature_columns ({len(cols)}) are not a registered feature set {sorted(FEATURE_SETS)}")


def column_index(columns) -> tuple[int, ...]:
    """Indices of a registered feature set's columns in FEATURE_COLUMNS (ValueError for an unregistered one)."""
    feature_set_name(columns)
    pos = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    return tuple(pos[c] for c in columns)
