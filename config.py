"""Global configuration values for Smart AgroSense (no growth stage)."""

# Default pump flow rate in litres per minute
PUMP_FLOW_RATE_LPM = 50.0

# Log file for all irrigation recommendations
IRRIGATION_LOG_FILE = "logs/irrigation_log.csv"

# Supported crops and their typical area (for default UI suggestions)
CROPS = [
    "tomato",
    "rice",
    "wheat",
    "maize",
    "potato",
]

DEFAULT_AREA_BY_CROP = {
    "tomato": 80.0,
    "rice": 120.0,
    "wheat": 100.0,
    "maize": 90.0,
    "potato": 70.0,
}
