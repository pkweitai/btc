import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Core paths/toggles
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", "./data"))
os.makedirs(DATA_DIR, exist_ok=True)

ENABLE_GOOGLE_TRENDS = os.getenv("ENABLE_GOOGLE_TRENDS", "0") == "1"
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "")

# --- NEW FLAGS (used by derivatives & market collectors) ---
# Skip Binance futures (geo-blocked in some regions)
SKIP_BINANCE_DERIVATIVES = os.getenv("SKIP_BINANCE_DERIVATIVES", "1") == "1"

# Use OKX as an additional derivatives source (no key needed)
USE_OKX_DERIVATIVES = os.getenv("USE_OKX_DERIVATIVES", "1") == "1"

# If you switched spot to binanceus, leave this True; otherwise ignore
USE_BINANCE_US_SPOT = os.getenv("USE_BINANCE_US_SPOT", "1") == "1"

SKIP_BYBIT_DERIVATIVES = os.getenv("SKIP_BYBIT_DERIVATIVES", "0") == "1"



