from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import tempfile
import chess
import chess.engine
import chess.pgn
from enum import Enum
from typing import List, Dict
import csv
import json

app = FastAPI()

# ---------------------------------------------------------------------------
# Enums and Analysis Parameters
# ---------------------------------------------------------------------------

class GamePhase(Enum):
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"

class Classification(Enum):
    BRILLIANT = "brilliant"
    GREAT = "great"
    BEST = "best"
    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    MISS = "miss"
    BLUNDER = "blunder"
    BOOK = "book"
    FORCED = "forced"

classification_values = {
    Classification.BLUNDER: 0,
    Classification.MISTAKE: 0.2,
    Classification.MISS: 0.3,
    Classification.INACCURACY: 0.4,
    Classification.GOOD: 0.65,
    Classification.EXCELLENT: 0.9,
    Classification.BEST: 1,
    Classification.GREAT: 1,
    Classification.BRILLIANT: 1,
    Classification.BOOK: 1,
    Classification.FORCED: 1,
}

centipawn_classifications = [
    Classification.BEST,
    Classification.EXCELLENT,
    Classification.GOOD,
    Classification.INACCURACY,
    Classification.MISS,
    Classification.MISTAKE,
    Classification.BLUNDER,
]

# Analysis parameters
FORCED_WIN_THRESHOLD = 500
MISS_CENTIPAWN_LOSS = 300
MISS_MATE_THRESHOLD = 3
ENDGAME_MATERIAL_THRESHOLD = 24
QUEEN_VALUE = 9

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def detect_game_phase(board: chess.Board, in_opening: bool) -> GamePhase:
    if in_opening:
        return GamePhase.OPENING

    total_material = 0
    queens = 0

    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in chess.PIECE_TYPES:
            if piece_type == chess.KING:
                continue
            count = len(board.pieces(piece_type, color))
            value = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: QUEEN_VALUE
            }[piece_type]
            total_material += count * value
            if piece_type == chess.QUEEN:
                queens += count

    endgame_conditions = [
        total_material <= ENDGAME_MATERIAL_THRESHOLD,
        queens == 0 and total_material <= ENDGAME_MATERIAL_THRESHOLD * 2,
    ]
    return GamePhase.ENDGAME if any(endgame_conditions) else GamePhase.MIDDLEGAME

def get_evaluation_loss_threshold(classif: Classification, prev_eval: float) -> float:
    prev_eval = abs(prev_eval)
    if classif == Classification.BEST:
        return max(0.0001 * prev_eval**2 + 0.0236 * prev_eval - 3.7143, 0)
    elif classif == Classification.EXCELLENT:
        return max(0.0002 * prev_eval**2 + 0.1231 * prev_eval + 27.5455, 0)
    elif classif == Classification.GOOD:
        return max(0.0002 * prev_eval**2 + 0.2643 * prev_eval + 60.5455, 0)
    elif classif == Classification.INACCURACY:
        return max(0.0002 * prev_eval**2 + 0.3624 * prev_eval + 108.0909, 0)
    elif classif == Classification.MISS:
        return max(0.00025 * prev_eval**2 + 0.38255 * prev_eval + 166.9541, 0)
    elif classif == Classification.MISTAKE:
        return max(0.0003 * prev_eval**2 + 0.4027 * prev_eval + 225.8182, 0)
    else:
        return float("inf")

def load_opening_book(csv_path: str) -> Dict[str, str]:
    opening_book = {}
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for row in reader:
                if len(row) < 3:
                    continue
                pgn_moves = row[2]
                game = chess.pgn.Game()
                board = game.board()
                for move in pgn_moves.split():
                    if "." in move:
                        continue
                    try:
                        chess_move = board.push_san(move)
                        fen = " ".join(board.fen().split()[:4])
                        opening_book[fen] = chess_move.uci()
                    except ValueError:
                        break
    except Exception as e:
        print(f"Error loading opening book: {e}")
    return opening_book

def is_book_move(board: chess.Board, opening_book: Dict[str, str], max_depth: int = 8) -> str:
    if board.fullmove_number > max_depth:
        return None
    fen = " ".join(board.fen().split()[:4])
    return opening_book.get(fen)

def get_phase_rating(classified_moves: List[Classification]) -> Classification:
    if not classified_moves:
        return Classification.GOOD
    total = sum(classification_values[m] for m in classified_moves)
    average = total / len(classified_moves)
    rating_order = [
        (Classification.BRILLIANT, 0.95),
        (Classification.GREAT, 0.85),
        (Classification.BEST, 0.75),
        (Classification.EXCELLENT, 0.65),
        (Classification.GOOD, 0.5),
        (Classification.INACCURACY, 0.35),
        (Classification.MISS, 0.25),
        (Classification.MISTAKE, 0.15)
    ]
    return next((c for c, t in rating_order if average >= t), Classification.BLUNDER)

# ---------------------------------------------------------------------------
# Analysis Function for JSON Games
# ---------------------------------------------------------------------------

def analyze_json_game_with_stockfish(game_json: Dict, engine_path: str, book_csv: str) -> Dict:
    """
    Analyze a chess game provided as a JSON object with metadata and a "Moves" list.
    Each move in the "Moves" list is a string (e.g., "1. Nf3 d6").
    """
    opening_book = load_opening_book(book_csv)
    board = chess.Board()
    result = {
        "move_analysis": [],
        "phase_analysis": {},
        "player_summaries": {},
        "game_info": game_json  # echoing back the game metadata
    }
    classifications = {
        "white": {phase: [] for phase in GamePhase},
        "black": {phase: [] for phase in GamePhase}
    }
    phase_data = {phase: [] for phase in GamePhase}
    in_opening = True
    move_number = 1

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        # Process each move line from the "Moves" list.
        # Each move line might contain one or two moves.
        for move_line in game_json.get("Moves", []):
            tokens = move_line.split()
            # Remove move number tokens (ending with a period)
            move_tokens = [token for token in tokens if not token.endswith('.')]
            for token in move_tokens:
                # Analyze position before making the move
                pre_info = engine.analyse(board, chess.engine.Limit(depth=20))
                pre_eval = pre_info["score"].white().score(mate_score=10000) or 0
                best_move = pre_info.get("pv", [None])[0]
                
                try:
                    board.push_san(token)
                except ValueError as e:
                    # Skip invalid moves with an error note
                    result["move_analysis"].append({
                        "move_number": move_number,
                        "player": "unknown",
                        "move": token,
                        "error": f"Invalid move: {e}"
                    })
                    continue

                # Analyze position after the move
                post_info = engine.analyse(board, chess.engine.Limit(depth=20))
                post_eval = post_info["score"].white().score(mate_score=10000) or 0

                # Determine game phase and opening book move
                book_move = is_book_move(board, opening_book)
                current_phase = detect_game_phase(board, in_opening)
                if not book_move and in_opening:
                    in_opening = False

                eval_loss = abs(pre_eval - post_eval)

                # Initial classification
                classification = Classification.BOOK if book_move else None
                if not classification:
                    for classif in centipawn_classifications:
                        threshold = get_evaluation_loss_threshold(classif, pre_eval)
                        if eval_loss <= threshold:
                            classification = classif
                            break
                    if classification is None:
                        classification = Classification.BLUNDER

                # Check for missed opportunities
                is_winning = abs(pre_eval) >= FORCED_WIN_THRESHOLD
                is_forced_win = pre_info["score"].is_mate() and pre_info["score"].relative.mate() <= MISS_MATE_THRESHOLD
                if is_winning and token != (best_move.uci() if best_move else None) and (eval_loss >= MISS_CENTIPAWN_LOSS or is_forced_win):
                    classification = Classification.MISS

                # Check for brilliant moves
                if classification == Classification.BEST:
                    if pre_eval < -150 and post_eval >= 150:
                        classification = Classification.GREAT
                    elif pre_eval < -300 and post_eval >= 300:
                        classification = Classification.BRILLIANT

                # Determine which player made the move (after push, board.turn points to opponent)
                player = "white" if board.turn == chess.BLACK else "black"
                classifications[player][current_phase].append(classification)
                phase_data[current_phase].append(classification)

                result["move_analysis"].append({
                    "move_number": move_number,
                    "player": "White" if player == "white" else "Black",
                    "move": token,
                    "evaluation": post_eval / 100,
                    "evaluation_loss": eval_loss / 100,
                    "classification": classification.value
                })
                move_number += 1

        # Phase analysis report
        for phase in GamePhase:
            moves = phase_data[phase]
            if moves:
                rating = get_phase_rating(moves)
                result["phase_analysis"][phase.value] = {
                    "rating": rating.value,
                    "move_count": len(moves)
                }

        # Player summaries report
        for color in ["white", "black"]:
            player_name = game_json.get("White") if color == "white" else game_json.get("Black")
            counts = {c.value: 0 for c in Classification}
            for phase in GamePhase:
                for m in classifications[color][phase]:
                    counts[m.value] += 1
            result["player_summaries"][player_name] = counts

    return result

# ---------------------------------------------------------------------------
# FastAPI Endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze-pgn/")
async def analyze_pgn(json_file: UploadFile = File(...), engine_path: str = "stockfish-windows-x86-64.exe", book_csv: str = "openings_master.csv"):
    try:
        # Save the uploaded JSON file to a temporary file
        contents = await json_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        # Load the JSON file
        with open(tmp_file_path, 'r', encoding="utf-8") as f:
            game_data = json.load(f)

        os.remove(tmp_file_path)

        # Check if the JSON file contains a game with a "Moves" list.
        if not isinstance(game_data, dict) or "Moves" not in game_data:
            raise HTTPException(status_code=400, detail="JSON file must contain game metadata and a 'Moves' list.")

        # Analyze the JSON game
        analysis_result = analyze_json_game_with_stockfish(game_data, engine_path, book_csv)
        return {"analysis_results": analysis_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Test Client Code (for local testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # For local testing, run the API on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)