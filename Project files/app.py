from flask import Flask, request, render_template
import os
import logging
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

try:
    model = load_model("vgg16_model.h5")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error("Error loading model", exc_info=True)
    model = None

butterfly_names = {
    0: "ADONIS",
    1: "AFRICAN GIANT SWALLOWTAIL",
    2: "AMERICAN SNOUT",
    3: "AN 88",
    4: "APPOLLO",
    5: "ATALA",
    6: "BANDED ORANGE HELICONIAN",
    7: "BANDED PEACOCK",
    8: "BECKER'S WHITE",
    9: "BLACK HAIRSTREAK",
    10: "BLUE MORPHO",
    11: "BLUE SPOTTED CHARAXES",
    12: "BROWN SIPROETA",
    13: "CABBAGE WHITE",
    14: "CAIRNS BIRDWING",
    15: "CHECQUERED SKIPPER",
    16: "CHESTNUT",
    17: "CLEOPATRA",
    18: "CLODIUS PARNASSIAN",
    19: "CLOUDED SULPHUR",
    20: "COMMON BANDED AWL",
    21: "COMMON WOOD-NYMPH",
    22: "COPPER TAIL",
    23: "CRESCENT",
    24: "CRIMSON PATCH",
    25: "DANAID EGGFLY",
    26: "EASTERN COMA",
    27: "EASTERN DAPPLE WHITE",
    28: "EASTERN PINE ELFIN",
    29: "ELBOWED PIERROT",
    30: "GOLD BANDED",
    31: "GREAT EGGFLY",
    32: "GREAT JAY",
    33: "GREEN CELLED CATTLEHEART",
    34: "GREY HAIRSTREAK",
    35: "INDRA SWALLOWTAIL",
    36: "ISHTAR SISTER",
    37: "JULIA",
    38: "LARGE MARBLE",
    39: "MALACHITE",
    40: "MANGROVE SKIPPER",
    41: "MESTRA",
    42: "METALMARK",
    43: "MILBERT'S TORTOISESHELL",
    44: "MONARCH",
    45: "MOURNING CLOAK",
    46: "ORANGE OAKLEAF",
    47: "ORANGE TIP",
    48: "ORCHARD SWALLOWTAIL",
    49: "PAINTED LADY",
    50: "PAPER KITE",
    51: "PEACOCK",
    52: "PINE WHITE",
    53: "PIPEVINE SWALLOWTAIL",
    54: "POPINJAY",
    55: "PURPLE HAIRSTREAK",
    56: "PURPLISH COPPER",
    57: "QUESTION MARK",
    58: "RED ADMIRAL",
    59: "RED CRACKER",
    60: "RED POSTMAN",
    61: "RED SPOTTED PURPLE",
    62: "SCARCE SWALLOWTAIL",
    63: "SILVER SPOT SKIPPER",
    64: "SLEEPY ORANGE",
    65: "SOOTYWING",
    66: "SOUTHERN DOGFACE",
    67: "STRAITED QUEEN",
    68: "TROPICAL LEAFWING",
    69: "TWO BARRED FLASHER",
    70: "ULYSES",
    71: "VICEROY",
    72: "WOOD SATYR",
    73: "YELLOW SWALLOWTAIL",
    74: "ZEBRA LONG WING"
}

UPLOAD_FOLDER = os.path.join("static", "images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def main_index():
    return render_template("index.html")

@app.route("/input")
def input_page():
    return render_template("input.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return render_template("output.html", prediction="No file uploaded", image_path="")

    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected")
        return render_template("output.html", prediction="No file selected", image_path="")

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        logging.debug(f"File saved to {filepath}")

        # Preprocess
        image = load_img(filepath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        # Predict
        preds = model.predict(image)
        pred_index = np.argmax(preds, axis=1)[0]
        prediction = butterfly_names.get(pred_index, "Unknown")

        return render_template("output.html", prediction=prediction, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
