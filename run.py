"""Application entry‑point compliant with Dash ≥2.16 (uses `app.run`)."""
import os

from app import create_app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    debug = bool(os.getenv("DEBUG", ""))

    app = create_app()
    # Dash ≥2.16 deprecates run_server → use run
    app.run(port=port, host="0.0.0.0", debug=debug)
