"""Generate a realistic synthetic dataset for crop irrigation (no growth stage).

Columns:
    crop            -> type of crop
    soil_moisture   -> percentage (10–90)
    temperature     -> °C (15–40)
    humidity        -> % (30–95)
    water_lpm2      -> target litres per m²
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CROPS = ["tomato", "rice", "wheat", "maize", "potato"]


def _base_water_need(crop: str) -> float:
    """Return a base litres/m² value depending on crop only."""
    # Rough relative needs
    crop_factor = {
        "rice": 1.3,
        "maize": 1.1,
        "tomato": 1.1,
        "potato": 0.9,
        "wheat": 0.8,
    }.get(crop, 1.0)

    # base between ~4 and 8 L/m²
    return 6.0 * crop_factor


def generate_dataset(n_samples: int = 400, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    crop = rng.choice(CROPS, size=n_samples)

    soil_moisture = rng.uniform(10, 90, size=n_samples)  # %
    temperature = rng.uniform(15, 40, size=n_samples)    # °C
    humidity = rng.uniform(30, 95, size=n_samples)       # %

    water_needed = []
    for c, sm, t, h in zip(crop, soil_moisture, temperature, humidity):
        base = _base_water_need(c)

        # moisture effect: dry (<30%) -> +40%, wet (>70%) -> -40%
        if sm < 30:
            moisture_factor = 1.4
        elif sm > 70:
            moisture_factor = 0.6
        else:
            moisture_factor = 1.0

        # temperature effect: hotter than 32C -> +20%, cooler than 20C -> -15%
        if t > 32:
            temp_factor = 1.2
        elif t < 20:
            temp_factor = 0.85
        else:
            temp_factor = 1.0

        # humidity effect: very humid (>80%) -> -10%, very dry (<40%) -> +10%
        if h > 80:
            hum_factor = 0.9
        elif h < 40:
            hum_factor = 1.1
        else:
            hum_factor = 1.0

        noise = rng.normal(0, 0.7)
        need = base * moisture_factor * temp_factor * hum_factor + noise
        need = max(0.0, need)  # no negative water
        water_needed.append(need)

    df = pd.DataFrame(
        {
            "crop": crop,
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "water_lpm2": water_needed,
        }
    )
    return df
