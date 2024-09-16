document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const resultsContainer = document.getElementById('resultsContainer');
    const statusMessage = document.getElementById('statusMessage');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = new FormData();
        formData.append('image', imageInput.files[0]);

        statusMessage.textContent = 'Processing your request...';
        resultsContainer.innerHTML = '';

        try {
            const response = await fetch('/process-address', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();
            if (data.error) {
                statusMessage.textContent = `Error: ${data.error}`;
            } else {
                statusMessage.textContent = 'Delivery information processed successfully!';
                displayResults(data);
            }
        } catch (error) {
            console.error('Error:', error);
            statusMessage.textContent = 'An error occurred while processing your request. Please try again.';
        }
    });

    function displayResults(data) {
        const extractedText = document.createElement('p');
        extractedText.innerHTML = `<strong>Extracted Text:</strong> ${data.extracted_text}`;
        resultsContainer.appendChild(extractedText);

        const predictedPin = document.createElement('p');
        predictedPin.innerHTML = `<strong>Predicted PIN Code:</strong> ${data.predicted_pin_code}`;
        resultsContainer.appendChild(predictedPin);

        const distance = document.createElement('p');
        distance.innerHTML = `<strong>Estimated Distance:</strong> ${data.distance}`;
        resultsContainer.appendChild(distance);

        const carbonFootprint = document.createElement('p');
        carbonFootprint.innerHTML = `<strong>Estimated Carbon Footprint:</strong> ${data.carbon_footprint_g}g CO2`;
        resultsContainer.appendChild(carbonFootprint);

        const trackingLink = document.createElement('p');
        trackingLink.innerHTML = `<strong>Tracking Link:</strong> <a href="${data.tracking_link}" target="_blank">${data.tracking_link}</a>`;
        resultsContainer.appendChild(trackingLink);

        const routeInfo = document.createElement('div');
        routeInfo.innerHTML = `<strong>Route Information:</strong><br>
            <strong>Distance:</strong> ${data.route_info.distance}<br>
            <strong>Duration:</strong> ${data.route_info.duration}<br>
            <strong>Steps:</strong> <ul>${data.route_info.steps.map(step => `<li>${step}</li>`).join('')}</ul>`;
        resultsContainer.appendChild(routeInfo);
    }
});
