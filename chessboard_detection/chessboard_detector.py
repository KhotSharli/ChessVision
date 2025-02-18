import os
from ultralytics import YOLO
import cv2

class ChessboardDetector:
    def __init__(self):
        # Load the trained models
        self.seg_model = YOLO('SegModel.pt')
        self.piece_model = YOLO('chessDetection3d.pt')
        self.class_map = {
            0: 'p', 1: 'r', 2: 'n', 3: 'b', 4: 'q', 5: 'k',
            6: 'P', 7: 'R', 8: 'N', 9: 'B', 10: 'Q', 11: 'K'
        }
        print("Models loaded successfully!")

    def detect_chessboards(self, image):
        """Detect all chessboards in the image."""
        results = self.seg_model(image)
        chessboards = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                chessboards.append((x1, y1, x2, y2))
        return chessboards

    def detect_chess_pieces(self, image):
        """Detect chess pieces in the image."""
        results = self.piece_model(image)
        pieces = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                pieces.append((x1, y1, x2, y2, cls, conf))
        return pieces

    def calculate_fen(self, chessboard_img, pieces):
        """Convert detected pieces to FEN notation with correct positioning."""
        height, width = chessboard_img.shape[:2]
        square_w = width / 8
        square_h = height / 8

        # Initialize empty board
        board = [['' for _ in range(8)] for _ in range(8)]

        for (x1, y1, x2, y2, cls, _) in pieces:
            # Get piece center coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Calculate board position
            file = int(center_x // square_w)  # File (column) from left to right (a to h)
            rank = int(center_y // square_h)  # Rank from top to bottom (0 to 7, where 0 is rank 8)

            # Ensure correct mapping to FEN
            board[rank][file] = self.class_map.get(cls, '')

        # Convert to FEN notation
        fen_rows = []
        for row in board:
            fen = ''
            empty = 0
            for cell in row:
                if cell:
                    if empty > 0:
                        fen += str(empty)
                        empty = 0
                    fen += cell
                else:
                    empty += 1
            if empty > 0:
                fen += str(empty)
            fen_rows.append(fen)

        return '/'.join(fen_rows) + ' w - - 0 1'

    def find_chessboard(self, image, click_coords):
        """Find the chessboard at the given click coordinates in the image."""
        x, y = click_coords
        chessboards = self.detect_chessboards(image)
        for (x1, y1, x2, y2) in chessboards:
            if x1 <= x <= x2 and y1 <= y <= y2:
                chessboard_crop = image[y1:y2, x1:x2]
                return chessboard_crop
        return None