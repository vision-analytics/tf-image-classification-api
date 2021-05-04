from datetime import datetime
import json
import logging
import os
import sys

from flask import Flask, request, jsonify, make_response, abort, Response

from modules import utilities
from modules.Classifier import Classifier

HTTP_STATUS_OK = 200
HTTP_STATUS_INVALID_INPUT = 400
HTTP_STATUS_INTERNAL_ERROR = 500

# READ CONFIG
PORT = int(utilities.get_config(section_name="FLASK", key='port'))

# SET LOGGING PARAMETERS
logging.basicConfig(filename=utilities.get_config("LOGGING", 'log_file_path'),
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    filemode='a')

# flask init
app  = Flask(__name__)

# classifier init
classifier = Classifier()

def parse_request(request):
    """Function to parse request data
    Args: 
        flask.request: Request
    Returns: 
        request.json.items : dict
    """
    print("parsing input data..")
    
    is_valid = False
    img = None

    if 'img' in request.json:
        logging.info("type: base64")
        img_b64 = request.json['img']

        img = utilities.convert_base64_to_image(img_b64)
        if img is not None: 
            is_valid = True
        return is_valid, img
    elif 'url' in request.json:
        logging.info("type: url")
        
        img_url = request.json['url']
        logging.info(f"url: {img_url}")

        try:
            img = utilities.read_image_from_url(img_url)
        except Exception as e:
            print(f"invalid url/image : {e}")
            logging.error(f"invalid url/image : {e}")
        
        if img is not None: 
            is_valid = True
        
        return is_valid, img
    else:
        print("bad request")
        logging.warning("bad request")
        abort(Response(
            "bad request",
            status=HTTP_STATUS_INVALID_INPUT,)
            )


@app.route("/api/v1/classify_image", methods=["POST"])
def classify_image():
    
    # create unique transaction_id
    transaction_id = utilities.create_transaction_id()
    print(f"***** Transaction id: {transaction_id}")
    logging.info(f"***** Transaction id: {transaction_id}")
    
    # parse request data
    is_valid, img = parse_request(request)

    # validate base64 (image)
    if not is_valid:
        print(f"{transaction_id} | unable to decode base64 (image)..")
        logging.warning(f"{transaction_id} | unable to decode base64 (image)..")
        result = {
            "message": "unable to decode base64 (image).", 
            "transaction_id": transaction_id
        }
        return make_response(jsonify(result), HTTP_STATUS_INVALID_INPUT, {'Content-Type': 'application/json'})

    
    # run prediction
    pred_result = classifier.predict(img)
    
    result = {
            "message": "Success!",
            "data": pred_result,
            "transaction_id": transaction_id
        }
    logging.info(f"{transaction_id} | {result}")
    logging.info(f"{transaction_id} ***** DONE!")
    return make_response(jsonify(result), HTTP_STATUS_OK, {'Content-Type': 'application/json'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=True, threaded=False)