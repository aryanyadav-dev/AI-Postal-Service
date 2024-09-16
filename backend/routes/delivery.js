const express = require('express');
const router = express.Router();
const { processAddress, getNearestPostOfficeRoute } = require('../services/deliveryService');

router.post('/process-address', async (req, res) => {
    try {
        if (!req.files || !req.files.image) {
            return res.status(400).json({ error: 'No image file provided' });
        }
        const imageFile = req.files.image.data; 
        const result = await processAddress(imageFile);
        res.json(result);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Failed to process address' });
    }
});

router.get('/nearest-post-office', async (req, res) => {
    try {
        const { origin, postOfficeType } = req.query;
        if (!origin || !postOfficeType) {
            return res.status(400).json({ error: 'Origin and post office type are required' });
        }
        const routeInfo = await getNearestPostOfficeRoute(origin, postOfficeType);
        res.json(routeInfo);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Failed to get route' });
    }
});

module.exports = router;
