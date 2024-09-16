const express = require('express');
const fileUpload = require('express-fileupload');
const cors = require('cors');
const dotenv = require('dotenv');
const path = require('path');
dotenv.config();

// Initialize Express app
const app = express();
app.use(cors());
app.use(express.json());
app.use(fileUpload());

// Static file serving for frontend assets (if applicable)
app.use(express.static(path.join(__dirname, 'public')));

// Import routes
const deliveryRoutes = require('./routes/delivery');
const userRoutes = require('./routes/user');

// Use routes with /api prefix
app.use('/api/delivery', deliveryRoutes);
app.use('/api/user', userRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
