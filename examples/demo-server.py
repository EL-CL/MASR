from flask import Flask, request

print("Loading model...")

import beamdecode

print("Model loaded")

app = Flask(__name__)


@app.route("/recognize", methods=["POST"])
def recognize():
    f = request.files["file"]
    f.save("test.wav")
    return beamdecode.predict("test.wav")


app.run("0.0.0.0", debug=True)