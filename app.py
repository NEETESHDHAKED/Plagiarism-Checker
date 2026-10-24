from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import easyocr
from PIL import Image

app = Flask(__name__)
print("Loading OCR ....")
reader = easyocr.Reader(['en','hi'])

def extract_text(file):
    filename = file.filename.lower()
    text = ""
    if file.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            t = page.extract_text()
            if t:
                text += t + " "
            elif filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                image_bytes = file.read()
                result = reader.readtext(image_bytes)
                text = " ".join([res[1] for res in result])
    else:
        text = file.read().decode('utf-8', errors='ignore')
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    status = None
    color = None
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1 and file2:
            text1 = extract_text(file1)
            text2 = extract_text(file2)

            if len(text1.strip()) < 10 or len(text2.strip()) < 10:
                result = "File me text bahut kam hai!"
            else:
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                percentage = round(similarity * 100, 2)

                result = f"{percentage}% Plagiarism"

                if percentage > 70:
                    status = "High Copy - Plagiarized"
                    color = "red"
                elif percentage > 40:
                    status = "Medium Copy - Partial Plagiarism"
                    color = "orange"
                else:
                    status = "Original - No Plagiarism"
                    color = "green"

    return render_template('index.html', result=result, status=status, color=color)

if __name__ == '__main__':
    app.run(debug=True)
