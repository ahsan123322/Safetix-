from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='notifications.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

@app.route('/api/notifications', methods=['POST'])
def receive_notification():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    detection_type = data.get('detection_type')
    confidence = data.get('confidence')
    
    if not detection_type or not confidence:
        return jsonify({"error": "Missing detection_type or confidence"}), 400
    
    # Log the notification
    logging.info(f"Received notification - Type: {detection_type}, Confidence: {confidence}")
    
    # You could add more processing here if needed
    
    return jsonify({"status": "Notification received successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)