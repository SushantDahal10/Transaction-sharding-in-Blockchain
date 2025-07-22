from flask import Flask
import os

app = Flask(__name__)

@app.route("/")
def ping():
    return "pong", 200

if __name__ == "__main__":
    # Use environment variable for port
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
