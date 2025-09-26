# Vietnamese Sentiment Analysis App

A lightweight full-stack application that serves PhoBERT-based sentiment analysis for Vietnamese text. The backend exposes a FastAPI endpoint backed by a fine-tuned transformer, while the React frontend delivers an interactive experience with confidence visualisations.

## Key Features
- Instant Vietnamese sentiment predictions (Negative, Neutral, Positive)
- Confidence percentage and class-wise probability breakdowns
- Curated example phrases for quick demos
- RESTful JSON API suitable for integration in other products

## Estimated Accuracy
The current PhoBERT checkpoint achieves an estimated accuracy of **70–85%** on internal validation samples. Performance may vary with domain-specific language or slang.

## Architecture
- **Backend:** FastAPI service loading a PhoBERT sequence classification model from `phobert_sentiment_model_final`. Torch handles inference, and CORS is enabled for local development.
- **Frontend:** React single-page app (`create-react-app`) that submits text to the API, shows loading states, and visualises prediction confidence.
- **Model Assets:** Tokeniser and weights shipped in `phobert_sentiment_model_final/` for offline inference.

## Prerequisites
- Python 3.10+ and pip
- Node.js 18+ and npm

Optional but recommended:
- A virtual environment for Python dependencies

## Backend Setup
1. Navigate to `backend/`.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Start the API server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

The server loads the PhoBERT model at startup and exposes `POST /predict` plus a health check at `/`.

## Frontend Setup
1. Navigate to `frontend/`.
2. Install dependencies: `npm install`
3. Ensure the API URL is set via `REACT_APP_API_URL` (defaults to `http://127.0.0.1:8000/predict`). Update `frontend/.env` if you run the backend elsewhere.
4. Run the development server:
   ```bash
   npm start
   ```

The app opens at `http://localhost:3000` and proxies sentiment requests to the API.

## API Reference
### `POST /predict`
Request body:
```json
{
  "text": "Sản phẩm tuyệt vời!"
}
```
Successful response:
```json
{
  "sentiment": "Positive",
  "confidence": 92.4,
  "probabilities": {
    "negative": 1.3,
    "neutral": 6.3,
    "positive": 92.4
  }
}
```
Errors return an `error` message suitable for displaying in the UI.

## Testing
- Frontend: `npm test` (uses React Testing Library).
- Backend: add FastAPI tests with tools such as `pytest` if needed.

## Deployment Notes
- Package the `phobert_sentiment_model_final/` directory with the backend service so weights are available at runtime.
- For production, restrict CORS to trusted origins and consider running `uvicorn` behind a process manager such as `gunicorn` or `uvicorn[standard]` workers.
