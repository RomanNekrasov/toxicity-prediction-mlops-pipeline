# importing Flask and other modules
from flask import Flask, jsonify, request
from component import *

def create_app():
    app = Flask(__name__)
    return app


# creating the app
app = create_app()

# A decorator used to tell the application which URL is associated function
# the complete URL will be http://ip:port/users?name=some_value
@app.route('/', methods=["POST"])
def clean_data():
    text = request.json.get('text')
    if not text:
        return jsonify(error="Please provide a 'text' field in the request body."), 400
    cleaned_text = clean_text(text)
    return jsonify(vector=str(cleaned_text)) # returning the cleaned text in json format


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))