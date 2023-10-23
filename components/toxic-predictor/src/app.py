# importing Flask and other modules
from flask import Flask, jsonify, request
from component import *

# Creating the app
app = Flask(__name__)


# A decorator used to tell the application which URL is associated function
# the complete URL will be http://ip:port/users?name=some_value
@app.route('/', methods=["POST"])
def do_something():
    pass


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))