const db = require('../config/db'); 

async function processAddress(imageFile) {
    try {
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
