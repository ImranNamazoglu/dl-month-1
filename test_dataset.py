import math
import numpy as np
from microgradimplengine import Value


def _values(row):
    return [Value(float(x)) for x in row]


def one_input_line_dataset():
    xs = [_values([i / 5.0]) for i in range(-20, 21)]
    ys = [2.0 * x[0].data + 1.0 for x in xs]
    return xs, ys


def two_input_plane_dataset():
    grid = [i / 2.0 for i in range(-4, 5)]
    xs = [_values([x1, x2]) for x1 in grid for x2 in grid]
    ys = [1.5 * x[0].data - 2.0 * x[1].data + 0.5 for x in xs]
    return xs, ys


def simple_curve_dataset():
    xs = [_values([i / 5.0]) for i in range(-20, 21)]
    ys = [x[0].data ** 2 for x in xs]
    return xs, ys


def shifted_quadratic_dataset():
    xs = [_values([i / 5.0]) for i in range(-20, 21)]
    ys = [0.5 * x[0].data ** 2 - x[0].data + 2.0 for x in xs]
    return xs, ys


def two_input_interaction_dataset():
    grid = [i / 2.0 for i in range(-4, 5)]
    xs = [_values([x1, x2]) for x1 in grid for x2 in grid]
    ys = [
        x[0].data * x[1].data + 0.5 * x[0].data - 1.5 * x[1].data + 1.0
        for x in xs
    ]
    return xs, ys


def smooth_wave_dataset():
    xs = [_values([i / 10.0]) for i in range(-40, 41)]
    ys = [math.sin(x[0].data) for x in xs]
    return xs, ys


def california_housing_feature_names():
    return [
        "bedrooms",
        "bathrooms",
        "sqft",
        "house_age_years",
        "distance_to_coast_miles",
        "school_score",
        "commute_minutes",
        "median_income_10k",
        "inventory_months",
    ]


def california_housing_feature_scales():
    return [5.0, 5.0, 4000.0, 100.0, 200.0, 10.0, 60.0, 30.0, 6.0]


def california_housing_target_scale():
    return 40.0


def california_housing_market_raw_dataset():
    # Representative 2025-2026 California housing examples for regression practice.
    # Target values are estimated sale prices in $100,000 units, not dollars.
    rows = [
        ("San Francisco condo", [2.0, 2.0, 1150.0, 35.0, 2.0, 8.5, 32.0, 16.0, 2.0], 12.8),
        ("San Francisco single family", [3.0, 2.0, 1650.0, 70.0, 3.0, 8.0, 38.0, 18.0, 1.8], 17.5),
        ("Palo Alto", [4.0, 3.0, 2400.0, 45.0, 8.0, 9.8, 28.0, 25.0, 1.5], 31.0),
        ("Mountain View", [3.0, 2.5, 1850.0, 35.0, 10.0, 9.2, 24.0, 22.0, 1.6], 21.5),
        ("Sunnyvale", [3.0, 2.0, 1600.0, 45.0, 13.0, 9.0, 25.0, 20.0, 1.7], 18.2),
        ("San Jose", [3.0, 2.0, 1550.0, 40.0, 18.0, 8.0, 30.0, 15.0, 2.1], 12.8),
        ("Santa Clara", [3.0, 2.5, 1700.0, 35.0, 15.0, 8.8, 26.0, 18.0, 1.8], 16.0),
        ("Fremont", [4.0, 2.5, 2100.0, 30.0, 20.0, 8.7, 35.0, 16.0, 2.0], 15.0),
        ("Oakland", [3.0, 2.0, 1450.0, 65.0, 5.0, 6.2, 35.0, 10.0, 2.5], 8.8),
        ("Berkeley", [3.0, 2.0, 1500.0, 75.0, 4.0, 8.2, 30.0, 12.0, 2.0], 13.0),
        ("Walnut Creek", [3.0, 2.0, 1800.0, 45.0, 25.0, 8.6, 35.0, 14.0, 2.2], 11.5),
        ("San Mateo", [3.0, 2.0, 1700.0, 55.0, 5.0, 9.0, 28.0, 20.0, 1.7], 19.0),
        ("Santa Cruz", [3.0, 2.0, 1550.0, 45.0, 1.0, 7.5, 32.0, 11.0, 2.0], 12.5),
        ("Sacramento", [3.0, 2.0, 1700.0, 25.0, 80.0, 6.8, 25.0, 8.0, 3.0], 5.3),
        ("Davis", [3.0, 2.0, 1650.0, 35.0, 75.0, 8.5, 18.0, 10.0, 2.2], 7.7),
        ("Roseville", [4.0, 3.0, 2400.0, 15.0, 95.0, 8.0, 30.0, 9.0, 3.0], 6.8),
        ("Stockton", [3.0, 2.0, 1600.0, 30.0, 60.0, 5.0, 35.0, 6.0, 4.0], 4.5),
        ("Modesto", [3.0, 2.0, 1550.0, 35.0, 75.0, 5.4, 32.0, 6.2, 3.8], 4.3),
        ("Fresno", [3.0, 2.0, 1650.0, 25.0, 130.0, 5.8, 24.0, 6.0, 4.0], 4.1),
        ("Clovis", [4.0, 2.5, 2200.0, 15.0, 135.0, 7.5, 25.0, 7.5, 3.2], 5.4),
        ("Bakersfield", [3.0, 2.0, 1750.0, 20.0, 105.0, 5.3, 22.0, 5.8, 4.5], 3.9),
        ("Merced", [3.0, 2.0, 1500.0, 20.0, 95.0, 5.2, 25.0, 5.4, 4.2], 3.8),
        ("Los Angeles", [3.0, 2.0, 1600.0, 60.0, 10.0, 6.5, 45.0, 10.0, 3.0], 10.5),
        ("Santa Monica", [3.0, 2.5, 1700.0, 55.0, 1.0, 8.8, 35.0, 17.0, 2.0], 22.0),
        ("Beverly Hills", [4.0, 4.0, 3200.0, 50.0, 8.0, 9.0, 35.0, 24.0, 2.0], 38.0),
        ("Pasadena", [3.0, 2.0, 1700.0, 70.0, 22.0, 7.8, 35.0, 11.0, 2.5], 11.8),
        ("Long Beach", [3.0, 2.0, 1450.0, 55.0, 3.0, 6.5, 35.0, 8.5, 3.0], 8.5),
        ("Torrance", [3.0, 2.0, 1600.0, 50.0, 5.0, 8.2, 32.0, 11.5, 2.4], 11.0),
        ("Irvine", [4.0, 3.0, 2400.0, 20.0, 10.0, 9.5, 28.0, 15.0, 2.2], 15.5),
        ("Anaheim", [3.0, 2.0, 1500.0, 45.0, 15.0, 6.8, 32.0, 8.0, 3.0], 8.6),
        ("Huntington Beach", [3.0, 2.0, 1650.0, 45.0, 2.0, 8.0, 32.0, 11.0, 2.4], 12.5),
        ("Newport Beach", [4.0, 3.5, 2800.0, 40.0, 1.0, 9.0, 30.0, 20.0, 1.8], 32.0),
        ("Riverside", [4.0, 2.5, 2200.0, 20.0, 45.0, 6.2, 40.0, 7.0, 4.0], 6.4),
        ("Corona", [4.0, 3.0, 2500.0, 18.0, 35.0, 7.0, 45.0, 8.5, 3.5], 7.6),
        ("San Bernardino", [3.0, 2.0, 1500.0, 35.0, 60.0, 4.8, 35.0, 5.2, 4.5], 4.6),
        ("Palm Springs", [3.0, 2.0, 1800.0, 35.0, 75.0, 6.5, 22.0, 7.0, 4.2], 6.2),
        ("San Diego", [3.0, 2.0, 1600.0, 45.0, 6.0, 7.5, 30.0, 10.0, 2.8], 9.5),
        ("La Jolla", [4.0, 3.0, 2600.0, 40.0, 1.0, 9.0, 25.0, 18.0, 2.0], 25.0),
        ("Carlsbad", [4.0, 3.0, 2400.0, 25.0, 3.0, 8.5, 30.0, 13.0, 2.5], 14.0),
        ("Chula Vista", [4.0, 2.5, 2100.0, 20.0, 8.0, 6.8, 32.0, 8.0, 3.2], 7.8),
        ("Oceanside", [3.0, 2.0, 1650.0, 35.0, 3.0, 6.8, 35.0, 7.8, 3.2], 8.0),
        ("Santa Barbara", [3.0, 2.0, 1700.0, 55.0, 2.0, 8.0, 22.0, 12.0, 2.0], 16.0),
        ("Monterey", [3.0, 2.0, 1550.0, 50.0, 2.0, 7.2, 25.0, 9.0, 2.5], 10.0),
        ("San Luis Obispo", [3.0, 2.0, 1600.0, 35.0, 10.0, 7.5, 18.0, 8.5, 2.5], 9.2),
        ("Napa", [3.0, 2.0, 1700.0, 35.0, 35.0, 7.0, 28.0, 9.5, 2.8], 8.8),
        ("Santa Rosa", [3.0, 2.0, 1650.0, 35.0, 25.0, 6.8, 28.0, 8.0, 3.0], 7.3),
        ("Redding", [3.0, 2.0, 1650.0, 30.0, 160.0, 5.8, 18.0, 5.6, 4.0], 3.9),
        ("Eureka", [3.0, 2.0, 1450.0, 60.0, 2.0, 5.8, 18.0, 5.5, 4.0], 4.2),
    ]
    xs = [_values(features) for _, features, _ in rows]
    ys = [target for _, _, target in rows]
    return xs, ys


def normalize_california_housing_features(features):
    scales = california_housing_feature_scales()
    return _values([feature / scale for feature, scale in zip(features, scales)])


def denormalize_california_housing_price(normalized_price):
    price_100k = normalized_price * california_housing_target_scale()
    return price_100k * 100000.0


def california_housing_market_dataset():
    raw_xs, raw_ys = california_housing_market_raw_dataset()
    scales = california_housing_feature_scales()
    target_scale = california_housing_target_scale()

    xs = [
        _values([feature.data / scale for feature, scale in zip(row, scales)])
        for row in raw_xs
    ]
    ys = [target / target_scale for target in raw_ys]
    return xs, ys


def california_housing_test_value():
    features = [3.0, 2.0, 1650.0, 40.0, 8.0, 7.5, 32.0, 10.0, 2.8]
    target = 9.5 / california_housing_target_scale()
    return normalize_california_housing_features(features), target


def california_housing_market_vector_dataset():
    xs, ys = california_housing_market_dataset()
    inputs = np.array([[feature.data for feature in row] for row in xs], dtype=float)
    targets = np.array(ys, dtype=float).reshape(-1, 1)
    return inputs, targets


def california_housing_vector_test_value():
    features, target = california_housing_test_value()
    inputs = np.array([[feature.data for feature in features]], dtype=float)
    targets = np.array([[target]], dtype=float)
    return inputs, targets


def tiny_regression_dataset():
    return two_input_plane_dataset()
