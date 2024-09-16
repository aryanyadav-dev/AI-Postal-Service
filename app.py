import os
import cv2
import pytesseract
import spacy
import numpy as np
import pandas as pd
import logging
import requests
import sqlite3
import geopy.distance  
from googletrans import Translator
from twilio.rest import Client
import pywhatkit
import openai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
translator = Translator()

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

        sample_address = pd.DataFrame({
            'locality': [address_entities.get('GPE', 'Unknown Locality')],
            'city': [address_entities.get('GPE', 'Unknown City')],
            'state': [address_entities.get('GPE', 'Unknown State')]
        })

        predicted_pin = '400001'  
        logger.info(f"Predicted PIN code: {predicted_pin}")

        origin = '19.0760,72.8777' 
        destination = address_entities.get('GPE', 'Unknown Location')  
        route_info = get_google_maps_route(origin, destination)
        if 'error' in route_info:
            return {'error': route_info['error']}

        distance = route_info['distance']
        carbon_footprint = calculate_carbon_footprint(float(distance.split()[0]))
        logger.info(f"Estimated distance: {distance}, Carbon footprint: {carbon_footprint}g CO2")

        customer_name = address_entities.get('PERSON', 'Unknown Customer')
        phone_number = '+91xxxxxxxxxx'  
        tracking_link = f"http://tracking_service/{predicted_pin}"  
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

def get_nearest_post_office_route(origin, post_office_type):
    try:
        post_office_locations = {
            'BPO': 'location_of_bpo',  
            'SPO': 'location_of_spo',  
            'HPO': 'location_of_hpo',  
            'GPO': 'location_of_gpo'   
        }

        destination = post_office_locations.get(post_office_type)
        if not destination:
            return {'error': 'Invalid post office type'}

        route_info = get_google_maps_route(origin, destination)
        if 'error' in route_info:
            return {'error': route_info['error']}

        return {
            'distance': route_info['distance'],
            'duration': route_info['duration'],
            'steps': route_info['steps']
        }
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {'error': 'An error occurred while fetching post office route'}

def calculate_carbon_footprint(distance):
    return distance * 0.5  

init_db()
