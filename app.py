import os
import logging
import sqlite3
import requests
import cv2
import pytesseract
import spacy
import numpy as np
import geopy
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from googletrans import Translator
from twilio.rest import Client
import pywhatkit as kit
import openai
import tensorflow as tf
import keras_ocr

load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
translator = Translator()

pipeline = keras_ocr.pipeline.Pipeline()

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')

if not account_sid or not auth_token or not twilio_phone_number:
    logger.error("Twilio credentials are not properly set in environment variables.")
    raise EnvironmentError("Twilio credentials are missing")

twilio_client = Client(account_sid, auth_token)

openai.api_key = os.getenv("OPENAI_API_KEY")

google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')

BACKEND_URL = 'http://localhost:5000'

post_office_locations = {
    'BPO': '19.0760,72.8777',  
    'SPO': '19.2183,72.9781', 
    'HPO': '18.9316,72.8333',  
    'GPO': '18.9397,72.8352'   
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
        status TEXT,
        transport_mode TEXT
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

# Function to get green routing based on transport mode
def get_green_route(mode, origin, destination):
    if mode == 'plane':
        # Example logic for plane route
        url = f"https://api.flightdata.com/routes?origin={origin}&destination={destination}&key={google_maps_api_key}"
    elif mode == 'ship':
        # Example logic for ship route
        url = f"https://api.shipdata.com/routes?origin={origin}&destination={destination}&key={google_maps_api_key}"
    elif mode == 'train':
        # Example logic for train route
        url = f"https://api.traindata.com/routes?origin={origin}&destination={destination}&key={google_maps_api_key}"
    elif mode == 'mail_van':
        # Example logic for mail van route
        url = f"https://api.trafficdata.com/routes?origin={origin}&destination={destination}&key={google_maps_api_key}"
    else:
        return {'error': 'Invalid transport mode'}

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

def process_address(image_file, use_tensorflow=False):
    try:
        # Read image file into numpy array
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Could not decode image")
            return {'error': 'Could not decode image'}

        extracted_text = ""
        if use_tensorflow:
            logger.info("Using TensorFlow and Keras-OCR for text extraction")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images = [image]
            prediction_groups = pipeline.recognize(images)
            extracted_text = ' '.join([text for text, box in prediction_groups[0]])
        else:
            logger.info("Using pytesseract for text extraction")
            extracted_text = pytesseract.image_to_string(image)

        logger.info(f"Extracted text: {extracted_text}")

        # Translate text if necessary
        translated_text = translator.translate(extracted_text).text
        logger.info(f"Translated text: {translated_text}")

        # Use NLP to extract entities
        doc = nlp(translated_text)
        address_entities = {ent.label_: ent.text for ent in doc.ents}
        logger.info(f"Extracted address entities: {address_entities}")

        customer_name = address_entities.get('PERSON', 'Unknown Customer')
        phone_number = '+91xxxxxxxxxx'  
        address = address_entities.get('GPE', 'Unknown Location')
        predicted_pin = '400001'  

        transport_mode = 'mail_van'  # Default transport mode, could be set based on input or logic
        origin = post_office_locations.get(transport_mode, '19.0760,72.8777')  # Default to BPO if mode not found
        route_info = get_green_route(transport_mode, origin, address)
        if 'error' in route_info:
            return {'error': route_info['error']}

        distance_str = route_info['distance']
        distance_value = float(distance_str.split()[0])
        carbon_footprint = calculate_carbon_footprint(distance_value)
        logger.info(f"Estimated distance: {distance_str}, Carbon footprint: {carbon_footprint}g CO2")

        tracking_link = f"http://tracking_service/{predicted_pin}"  
        status = 'In Progress'

        delivery_data = {
            'customer_name': customer_name,
            'phone_number': phone_number,
            'address': translated_text,
            'pin_code': predicted_pin,
            'tracking_link': tracking_link,
            'carbon_footprint': carbon_footprint,
            'status': status,
            'transport_mode': transport_mode
        }
        save_delivery_to_backend(delivery_data)
        logger.info("Delivery information saved to backend")

        # Send tracking link and carbon footprint via Twilio SMS
        message_body = f"Your delivery to {translated_text} with PIN {predicted_pin} is on its way. Track here: {tracking_link}. Estimated carbon footprint: {carbon_footprint}g CO2."
        twilio_client.messages.create(
            body=message_body,
            from_=twilio_phone_number,
            to=phone_number
        )
        logger.info("Tracking link sent via Twilio")

        # Send WhatsApp message using pywhatkit
        kit.sendwhatmsg_instantly(phone_number, f"Tracking details: {message_body}", wait_time=20)
        logger.info("WhatsApp message sent")

        feedback_prompt = f"Customer delivery to {translated_text} was successful. Please provide feedback."
        openai_response = openai.Completion.create(
            model="text-davinci-003",
            prompt=feedback_prompt,
            max_tokens=50
        )
        feedback_request = openai_response.choices[0].text.strip()
        logger.info(f"Generated feedback request: {feedback_request}")

        # Send feedback request via Twilio SMS
        feedback_message_body = f"Dear {customer_name}, we would appreciate your feedback on your recent delivery: {feedback_request}"
        twilio_client.messages.create(
            body=feedback_message_body,
            from_=twilio_phone_number,
            to=phone_number
        )
        logger.info("Feedback request sent via Twilio")

        # Send feedback request via WhatsApp
        kit.sendwhatmsg_instantly(phone_number, feedback_message_body, wait_time=20)
        logger.info("Feedback request sent via WhatsApp")

        return {
            'extracted_text': translated_text,
            'address_entities': address_entities,
            'predicted_pin_code': predicted_pin,
            'distance': distance_str,
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
    use_tensorflow = request.form.get('use_tensorflow', 'false').lower() == 'true'
    result = process_address(image_file, use_tensorflow)
    return jsonify(result)

def calculate_carbon_footprint(distance_km):
    # Example calculation: 150g CO2 per km 
    return distance_km * 150

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
