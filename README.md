# AI Post Office Identification System

## Overview

The AI Post Office Identification System is designed to automate the processing of delivery requests based on address images. It leverages Optical Character Recognition (OCR) and Natural Language Processing (NLP) to extract and interpret address information, predict PIN codes, calculate carbon footprints, and integrate with external services for notifications and feedback. Also designed a Landing page for a website for scanning purposes and other additional information using HTML,CSS and GSAP.

## Key Components

1. **Image Processing**:
   - **OpenCV**: Used to convert the image to a format suitable for OCR. The image is first converted to grayscale and thresholding is applied to improve OCR accuracy.
   - **Tesseract OCR**: Extracts text from the processed image.

2. **Text Translation and NLP**:
   - **Google Translator**: Translates the extracted text if needed.
   - **SpaCy**: Processes the translated text to extract address-related entities such as locality, city, and state.

3. **PIN Code Prediction**:
   - **Machine Learning Model**: A pre-trained model predicts the PIN code based on the extracted address information. The model is trained using features derived from address entities.

4. **Green Routing**:
   - **Google Maps API**: Calculates the optimal route from the origin to the destination, including distance and estimated time. Provides a green and traffic-free route to minimize carbon footprint.
   - **Carbon Footprint Calculation**: Estimates the carbon footprint based on the distance of the delivery.

5. **Database Management**:
   - **SQLite**: Stores delivery details including customer information, address, PIN code, tracking link, carbon footprint, and status.

6. **Notification and Feedback**:
   - **Twilio**: Sends SMS notifications with tracking information and feedback requests.
   - **Pywhatkit**: Integrates with WhatsApp to send tracking details and feedback requests.
   - **OpenAI**: Generates feedback requests using AI-based language models.

## Workflow

1. **Image Upload**: The user uploads an image of the address.
2. **OCR Processing**: The image is processed to extract text using Tesseract OCR.
3. **Text Translation and NLP**: The text is translated (if needed) and processed to identify address components.
4. **PIN Code Prediction**: The model predicts the PIN code based on the address.
5. **Route Calculation**: The optimal route is determined using the Google Maps API, and the carbon footprint is calculated.
6. **Data Storage**: Delivery details are saved in the SQLite database.
7. **Notifications**: Tracking information and feedback requests are sent via SMS and WhatsApp.
8. **Response**: A JSON response is returned with all relevant details, including the extracted text, predicted PIN code, route information, and notification statuses.

## Integration Points

- **Twilio**: For sending SMS notifications.
- **Pywhatkit**: For WhatsApp messaging.
- **OpenAI**: For generating feedback requests.
- **Google Maps API**: For route calculation.

## Setup

1. **API Keys**: Obtain and configure API keys for Twilio, OpenAI, and Google Maps.
2. **Dependencies**: Install required libraries including OpenCV, Tesseract, SpaCy, and Flask.
3. **Database**: Initialize and set up the SQLite database to store delivery information.

This application provides an automated and integrated solution for managing delivery requests, enhancing efficiency, and improving user experience through real-time notifications and accurate address processing.
