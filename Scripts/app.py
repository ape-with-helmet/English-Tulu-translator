from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('translation_model.h5')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    english_sentence = data['sentence']
    # Preprocess and predict
    # ...
    return jsonify({'tulu_translation': tulu_sentence})

if __name__ == '__main__':
    app.run(debug=True)
