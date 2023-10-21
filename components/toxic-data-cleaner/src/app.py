# importing Flask and other modules
from flask import Flask, request
from component import *
import pandas as pd

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application which URL is associated function
# the complete URL will be http://ip:port/users?name=some_value
@app.route('/', methods=["POST"])
def say_hello():
    text = request.data
    cleaned_text = clean_data(text)
    pd.Series()

    return "Hello " + name_value


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)