import cv2
import pytesseract
import spacy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
import logging
import geopy.distance
from googletrans import Translator
from twilio.rest import Client
import pywhatkit
import openai
import sqlite3
import requests


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
translator = Translator()

account_sid = 'your_twilio_account_sid'
auth_token = 'your_twilio_auth_token'
twilio_client = Client(account_sid, auth_token)
twilio_phone_number = 'your_twilio_phone_number'

openai.api_key = "your_openai_api_key"

google_maps_api_key = 'your_google_maps_api_key'

DATABASE = 'delivery_system.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
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
    conn.commit()
    conn.close()

def save_delivery(customer_name, phone_number, address, pin_code, tracking_link, carbon_footprint, status):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO deliveries (customer_name, phone_number, address, pin_code, tracking_link, carbon_footprint_g, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (customer_name, phone_number, address, pin_code, tracking_link, carbon_footprint, status))
    conn.commit()
    conn.close()

def calculate_carbon_footprint(distance_km):
    return distance_km * 150  # Assuming 150g CO2/km for delivery vehicles

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

@app.route('/process-address', methods=['POST'])
def process_address():
    try:
        if 'image' not in request.files:
            logger.error("No image file part in the request")
            return jsonify({'error': 'No image file part in the request'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            logger.error("No image selected for uploading")
            return jsonify({'error': 'No image selected for uploading'}), 400

        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        if image is None:
            logger.error("Could not decode image")
            return jsonify({'error': 'Could not decode image'}), 400

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

        sample_address_encoded = pd.get_dummies(sample_address).reindex(columns=X.columns, fill_value=0)
        predicted_pin = model.predict(sample_address_encoded)[0]

        logger.info(f"Predicted PIN code: {predicted_pin}")

        origin = '19.0760,72.8777'  
        destination = address_entities.get('GPE', 'Unknown Location')  
        route_info = get_google_maps_route(origin, destination)
        if 'error' in route_info:
            return jsonify({'error': route_info['error']}), 500

        distance = route_info['distance']
        carbon_footprint = calculate_carbon_footprint(float(distance.split()[0]))
        logger.info(f"Estimated distance: {distance}, Carbon footprint: {carbon_footprint}g CO2")

        customer_name = address_entities.get('PERSON', 'Unknown Customer')
        phone_number = '+91xxxxxxxxxx'  
        tracking_link = f"http://tracking_service/{predicted_pin}"  
        status = 'In Progress'
        save_delivery(customer_name, phone_number, translated_text, predicted_pin, tracking_link, carbon_footprint, status)
        logger.info("Delivery information saved to database")

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

        return jsonify({
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
        })

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred while processing the request'}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
