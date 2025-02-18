from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image
from chessboard_detection.chessboard_detector import ChessboardDetector

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'STATIC_FOLDER': 'static',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024 
})

for folder in [app.config['UPLOAD_FOLDER'], app.config['STATIC_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

chess_detector = ChessboardDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type"}), 400

    # Read PDF content into memory
    pdf_data = file.read()

    previews = []
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            
            # Convert to PIL Image
            img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert to base64
            buffer = BytesIO()
            img_pil.save(buffer, format="JPEG")
            base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            previews.append({
                "preview_data": f"data:image/jpeg;base64,{base64_img}",
                "page": page_num,
                "original_width": pix.width,
                "original_height": pix.height
            })
        return jsonify({"previews": previews})
    except Exception as e:
        return jsonify({"error": f"PDF processing failed: {str(e)}"}), 500

@app.route('/analyze', methods=['POST'])
def handle_analysis():
    data = request.json
    if 'image' not in data or 'origX' not in data or 'origY' not in data:
        return jsonify({"error": "Missing parameters: image, origX, and origY are required"}), 400

    try:
        # Ensure 'image' is a string
        if not isinstance(data['image'], str):
            return jsonify({"error": "Invalid image data format: expected base64 string"}), 400

        # Extract base64 image data
        header, encoded = data['image'].split(",", 1)
        image_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Get click coordinates
        x = int(data['origX'])
        y = int(data['origY'])

        # Detect chessboard at clicked location
        chessboard_crop = chess_detector.find_chessboard(image, (x, y))
        if chessboard_crop is None:
            return jsonify({"error": "No chessboard detected at the specified location"}), 404

        # Detect pieces and generate FEN
        pieces = chess_detector.detect_chess_pieces(chessboard_crop)
        fen = chess_detector.calculate_fen(chessboard_crop, pieces)

        # Convert chessboard crop to base64 for preview
        _, buffer = cv2.imencode('.jpg', chessboard_crop)
        chessboard_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "fen": fen,
            "chessboard_url": f"data:image/jpeg;base64,{chessboard_b64}"
        })

    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    
@app.route('/start_game', methods=['POST'])
def start_game():
    data = request.json
    required_fields = ['site', 'fen']
    if any(field not in data for field in required_fields):
        return jsonify({"error": "Missing required parameters"}), 400

    site = data['site']
    fen = data['fen']

    if site == 'lichess':
        url = f"https://lichess.org/editor/{fen}"
    elif site == 'chess.com':
        url = f"https://www.chess.com/analysis?fen={fen}"
    else:
        return jsonify({"error": "Invalid site"}), 400

    return jsonify({"redirect_url": url})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)