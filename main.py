from pathlib import Path
import os
import sys

from streamlit.web import cli as stcli


def resource_path(filename: str) -> Path:
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_dir / filename


def main() -> int:
    app_path = resource_path("app.py")

    if not app_path.exists():
        raise FileNotFoundError(f"Could not find Streamlit app at {app_path}")

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--server.fileWatcherType=none",
        *sys.argv[1:],
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
