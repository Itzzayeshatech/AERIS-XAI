"""Script to launch the AERIS XAI Dash frontend command centre."""
from backend.config import settings
from frontend.app import app

if __name__ == "__main__":
    print(f"Launching AERIS XAI Dash Interface on http://{settings.DASH_HOST}:{settings.DASH_PORT} ...")
    app.run(
        host=settings.DASH_HOST,
        port=settings.DASH_PORT,
        debug=False,
    )
