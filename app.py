import os
import runpy
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
APP_PATH = BASE_DIR / "5.Project Development Phase" / "app.py"


module_globals = runpy.run_path(str(APP_PATH), run_name="credit_card_app")
app = module_globals["app"]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)