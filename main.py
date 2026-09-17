import os

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    if debug:
        # Dev mode: auto-reload + interactive debugger. threaded=True keeps
        # one slow request (e.g. online translation) from blocking others.
        app.run(debug=True, threaded=True, port=port)
    else:
        # Production-safe default: no reloader/debugger, and waitress (unlike
        # Flask's built-in server) is meant to actually take real traffic.
        from waitress import serve

        serve(app, host="0.0.0.0", port=port)
