from flask import Flask, render_template, request, jsonify
import pickle
from flask_cors import CORS
import os
import webbrowser
from threading import Timer

# Đặt tên thư mục mới cho templates và static
custom_template_folder = 'Design/templates'
custom_static_folder = 'Design/static'  # Đặt tên thư mục tĩnh mới

app = Flask(__name__, template_folder=custom_template_folder, static_folder=custom_static_folder)
CORS(app)

with open('temp/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('temp/vectorized.pkl', 'rb') as f:
    vectorized = pickle.load(f)

@app.route('/')
def index():
    # Sử dụng os.path.join để kết hợp đường dẫn đến template
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    if text is not None:
        text_transformed = vectorized.transform([text])

        prediction = svm_model.predict(text_transformed)[0]

        return jsonify({'prediction': int(prediction)})
    else:
        return jsonify({'error': 'Input text not provided.'})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        Timer(1, open_browser).start()
    app.run(host='0.0.0.0', debug=True)
