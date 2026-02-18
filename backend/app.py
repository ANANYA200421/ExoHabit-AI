from flask import Flask, send_from_directory
from routes import predict_bp
import os

BASE = os.path.dirname(os.path.abspath(__file__))
FRONT = os.path.join(BASE, "../frontend")

app = Flask(__name__)
app.register_blueprint(predict_bp)

@app.route("/predict")
def predict_page():
    return send_from_directory(FRONT, "predict.html")

@app.route("/dashboard")
def dashboard_page():
    return send_from_directory(FRONT, "dashboard.html")

@app.route("/mission")
def mission_page():
    return send_from_directory(FRONT, "mission.html")

@app.route("/css/<path:p>")
def css(p):
    return send_from_directory(os.path.join(FRONT,"css"), p)

@app.route("/js/<path:p>")
def js(p):
    return send_from_directory(os.path.join(FRONT,"js"), p)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
