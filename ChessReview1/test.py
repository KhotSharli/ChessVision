import requests
import json

def test_analyze_pgn():
    url = "http://127.0.0.1:8000/analyze-pgn/"
    
    # Load the JSON file containing PGNs
    with open('game2.json', 'r') as f:
        pgn_data = json.load(f)
    
    # Send the JSON file to the endpoint
    files = {'json_file': ('game2.json', json.dumps(pgn_data), 'application/json')}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        analysis_result = response.json()

        # Save result to JSON file
        json_filename = "analysis_results.json"
        with open(json_filename, "w") as json_file:
            json.dump(analysis_result, json_file, indent=4)

        print(f"Test passed! Results saved to {json_filename}")
    else:
        print("Test failed! Status code:", response.status_code, "Response:", response.text)

if __name__ == "__main__":
    test_analyze_pgn()