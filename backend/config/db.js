const db = require('../config/db'); // Import the database connection

// Function to process delivery from address image
async function processAddress(imageFile) {
    try {
        // Since processAddress in Python handles file processing and OCR, this would typically involve:
        // 1. Upload the image file to a server or temporary storage
        // 2. Make a call to a Python backend or use a library to process the image
        // This code assumes you are directly interacting with a Python service for address processing.
        // For demonstration, we're returning a placeholder response here.

        const response = await fetch('http://localhost:5000/process-address', {
            method: 'POST',
            body: imageFile,
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error processing address:', error);
        throw new Error('Failed to process address');
    }
}

// Function to get nearest post office routing
async function getNearestPostOfficeRoute(origin, postOfficeType) {
    try {
        const response = await fetch(`http://localhost:5000/nearest-post-office?origin=${encodeURIComponent(origin)}&postOfficeType=${encodeURIComponent(postOfficeType)}`);
        const routeInfo = await response.json();
        return routeInfo;
    } catch (error) {
        console.error('Error getting nearest post office route:', error);
        throw new Error('Failed to get route');
    }
}

module.exports = {
    processAddress,
    getNearestPostOfficeRoute
};
