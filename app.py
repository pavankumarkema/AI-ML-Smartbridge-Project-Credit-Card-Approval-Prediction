from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent
APP_FILE = BASE_DIR / "5.Project Development Phase" / "app.py"


def load_app_module():
    spec = spec_from_file_location("credit_card_app", APP_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load app module from {APP_FILE}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_app_module = load_app_module()
app = _app_module.app
application = app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)