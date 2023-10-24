# importing Flask and other modules
from flask import Flask, jsonify, request
import logging
import sys
from component import *

# creating the app
app = Flask(__name__)


# A decorator used to tell the application which URL is associated function
# the complete URL will be http://ip:port/users?name=some_value
@app.route('/', methods=["POST"])
def clean_data():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    text = request.json.get('text')
    if not text:
        return jsonify(error="Please provide a 'text' field in the request body."), 400
    cleaned_text = clean_text(text)
    logging.info('...Cleaner api cleaned the text...')
    return jsonify(clean_text=str(cleaned_text))  # returning the cleaned text in json format


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
