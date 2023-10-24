from flask import request, render_template
import os

predictor_api_url = os.environ['PREDICTOR_API_URL']
