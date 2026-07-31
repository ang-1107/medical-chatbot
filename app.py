import os

from src.webapp import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=False)
