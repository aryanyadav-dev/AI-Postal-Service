const twilio = require('twilio');

const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;
const twilioPhoneNumber = process.env.TWILIO_PHONE_NUMBER;

if (!accountSid || !authToken || !twilioPhoneNumber) {
    throw new Error("Twilio credentials are not set in the environment variables.");
}

const client = twilio(accountSid, authToken);

async function sendSMS(to, body) {
    try {
        const message = await client.messages.create({
            body: body,
            from: twilioPhoneNumber,
            to: to
        });
        return message;
    } catch (error) {
        console.error('Error sending SMS:', error);
        throw new Error('Failed to send SMS');
    }
}

module.exports = {
    client,
    sendSMS
};
