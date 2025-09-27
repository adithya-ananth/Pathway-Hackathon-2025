# Pathfinders Submission for Pathway Hackathon 2025

## Running locally:

### Clone the repository
```
git clone git@github.com:adithya-ananth/Pathway-Hackathon-2025.git
```
### Add .env file in backend directory with the following content: (3 distinct keys needed)
```
GEMINI_API_KEY_EXTRACT=your_key_here
GEMINI_API_KEY_ENRICH=your_key_here
GEMINI_API_KEY_SUMMARIZE=your_key_here
```
### Run the backend
```
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8080
```
### Run the frontend
```
cd frontend
npm install
npm run dev
```