from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from .analysis import analyze_chart
from .model import update_feedback, load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('chart')
        strategy = request.form.get('strategy')
        if file and strategy:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = analyze_chart(filepath, strategy, MODEL_PATH)
            return render_template('result.html', result=result)
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    recommendation = request.form.get('recommendation')
    correct = request.form.get('correct') == 'yes'
    update_feedback(recommendation, correct, MODEL_PATH)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
