import os
import logging
import sqlite3
import requests
import cv2
import pytesseract
import spacy
import numpy as np
import pandas as pd
import geopy
from flask import Flask, request, jsonify, redirect, url_for
from dotenv import load_dotenv
from googletrans import Translator
from twilio.rest import Client
import pywhatkit
import openai

# Load environment variables
load_dotenv()

# Flask app initialization
app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and APIs
nlp = spacy.load("en_core_web_sm")
translator = Translator()

# Twilio setup
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')

if not account_sid or not auth_token or not twilio_phone_number:
    logger.error("Twilio credentials are not properly set in environment variables.")
    raise EnvironmentError("Twilio credentials are missing")

twilio_client = Client(account_sid, auth_token)

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Google Maps API key
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')

# Backend URL
BACKEND_URL = 'http://localhost:5000'

# Post office locations
post_office_locations = {
    'BPO': '19.0760,72.8777',  # Example coordinates for BPO in Mumbai
    'SPO': '19.2183,72.9781',  # Example coordinates for SPO in Mumbai
    'HPO': '18.9316,72.8333',  # Example coordinates for HPO in Mumbai
    'GPO': '18.9397,72.8352'   # Example coordinates for GPO in Mumbai
}

# Database initialization
def init_db():
    db_path = 'delivery_system.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS deliveries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT,
        phone_number TEXT,
        address TEXT,
        pin_code INTEGER,
        tracking_link TEXT,
        carbon_footprint_g REAL,
        status TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT
    )
    ''')

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Helper functions to interact with backend API
def save_user_to_backend(username, password):
    response = requests.post(f'{BACKEND_URL}/user/register', json={'username': username, 'password': password})
    return response.json()

def verify_user_from_backend(username, password):
    response = requests.post(f'{BACKEND_URL}/user/login', json={'username': username, 'password': password})
    return response.json()

def save_delivery_to_backend(data):
    response = requests.post(f'{BACKEND_URL}/delivery/save', json=data)
    return response.json()

def get_google_maps_route(origin, destination):
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={google_maps_api_key}&traffic_model=best_guess"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        route = data['routes'][0]['legs'][0]
        distance = route['distance']['text']
        duration = route['duration']['text']
        steps = [step['html_instructions'] for step in route['steps']]
        return {
            'distance': distance,
            'duration': duration,
            'steps': steps
        }
    else:
        return {'error': 'Unable to fetch route'}

# Function to process address from image
def process_address(image_file):
    try:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.error("Could not decode image")
            return {'error': 'Could not decode image'}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        text = pytesseract.image_to_string(thresh, lang='eng')
        logger.info(f"Extracted text: {text}")

        translated_text = translator.translate(text).text
        logger.info(f"Translated text: {translated_text}")

        doc = nlp(translated_text)
        address_entities = {ent.label_: ent.text for ent in doc.ents}
        logger.info(f"Extracted address entities: {address_entities}")

        customer_name = address_entities.get('PERSON', 'Unknown Customer')
        phone_number = '+91xxxxxxxxxx'  # Example phone number
        address = address_entities.get('GPE', 'Unknown Location')
        predicted_pin = '400001'  # For demonstration

        origin = '19.0760,72.8777'  # Example origin in Mumbai
        destination = address
        route_info = get_google_maps_route(origin, destination)
        if 'error' in route_info:
            return {'error': route_info['error']}

        distance = route_info['distance']
        carbon_footprint = calculate_carbon_footprint(float(distance.split()[0]))
        logger.info(f"Estimated distance: {distance}, Carbon footprint: {carbon_footprint}g CO2")

        tracking_link = f"http://tracking_service/{predicted_pin}"  # Example tracking link
        status = 'In Progress'

        delivery_data = {
            'customer_name': customer_name,
            'phone_number': phone_number,
            'address': translated_text,
            'pin_code': predicted_pin,
            'tracking_link': tracking_link,
            'carbon_footprint': carbon_footprint,
            'status': status
        }
        save_delivery_to_backend(delivery_data)
        logger.info("Delivery information saved to backend")

        message_body = f"Your delivery to {translated_text} with PIN {predicted_pin} is on its way. Track here: {tracking_link}. Estimated carbon footprint: {carbon_footprint}g CO2."
        twilio_client.messages.create(
            body=message_body,
            from_=twilio_phone_number,
            to=phone_number
        )
        logger.info("Tracking link sent via Twilio")

        pywhatkit.sendwhatmsg_instantly(phone_number, f"Tracking details: {message_body}", wait_time=20)
        logger.info("WhatsApp message sent")

        feedback_prompt = f"Customer delivery to {translated_text} was successful. Please provide feedback."
        openai_response = openai.Completion.create(
            model="text-davinci-003",
            prompt=feedback_prompt,
            max_tokens=50
        )
        feedback_request = openai_response.choices[0].text.strip()
        logger.info(f"Generated feedback request: {feedback_request}")

        feedback_message_body = f"Dear {customer_name}, we would appreciate your feedback on your recent delivery: {feedback_request}"
        twilio_client.messages.create(
            body=feedback_message_body,
            from_=twilio_phone_number,
            to=phone_number
        )
        logger.info("Feedback request sent via Twilio")

        pywhatkit.sendwhatmsg_instantly(phone_number, feedback_message_body, wait_time=20)
        logger.info("Feedback request sent via WhatsApp")

        return {
            'extracted_text': translated_text,
            'address_entities': address_entities,
            'predicted_pin_code': predicted_pin,
            'distance': distance,
            'carbon_footprint_g': carbon_footprint,
            'tracking_link': tracking_link,
            'notification_status': 'sent',
            'whatsapp_status': 'sent',
            'feedback_request': feedback_request,
            'route_info': route_info
        }
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {'error': 'An error occurred while processing the request'}

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    result = process_address(image_file)

    if 'error' in result:
        return jsonify(result), 500

    return jsonify(result)

# Routes to handle post office selection
@app.route('/post_office_options')
def show_post_office_options():
    origin = request.args.get('origin')
    if not origin:
        return jsonify({'error': 'Origin not provided'}), 400

    return f'''
    <html>
    <head>
        <title>Select Nearest Post Office</title>
    </head>
    <body>
        <h1>Select Your Nearest Post Office</h1>
        <form method="POST" action="/route_to_post_office">
            <input type="hidden" name="origin" value="{origin}">
            <button type="submit" name="po_type" value="BPO">Nearest BPO</button>
            <button type="submit" name="po_type" value="SPO">Nearest SPO</button>
            <button type="submit" name="po_type" value="HPO">Nearest HPO</button>
            <button type="submit" name="po_type" value="GPO">Nearest GPO</button>
        </form>
    </body>
    </html>
    '''

@app.route('/route_to_post_office', methods=['POST'])
def route_to_post_office():
    origin = request.form.get('origin')
    po_type = request.form.get('po_type')

    destination = post_office_locations.get(po_type)
    if not destination:
        return jsonify({'error': 'Invalid post office type'}), 400

    route_info = get_google_maps_route(origin, destination)
    if 'error' in route_info:
        return jsonify({'error': route_info['error']}), 500

    return jsonify({
        'origin': origin,
        'destination': destination,
        'route_info': route_info
    })

# Carbon footprint calculation
def calculate_carbon_footprint(distance_km):
    emission_factor_g_per_km = 150  # Example value
    return distance_km * emission_factor_g_per_km

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
