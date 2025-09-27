# Pathfinders Submission for Pathway Hackathon 2025

## Steps to run locally:

### 1. Clone the repository
```
git clone git@github.com:adithya-ananth/Pathway-Hackathon-2025.git
```
### 2. Add .env file in backend directory with the following content: (3 distinct keys needed)
```
GEMINI_API_KEY_EXTRACT=your_key_here
GEMINI_API_KEY_ENRICH=your_key_here
GEMINI_API_KEY_SUMMARIZE=your_key_here
```
### 3. Set up virtual environment
```
python -m venv env
source env/bin/activate
```
### 4. Install the required libraries from the requirements.txt file using pip
```
pip install -r requirements.txt
```
### 5. Run the backend
```
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8080
```
### 6. Run the frontend
```
cd frontend
npm install
npm run dev
```
### Project Description:
Our application provides Real-time Academic Research Discovery with AI-Powered Fallback, to provide accurate and efficient replies to research-related user queries. Using Pathway's parser and RAG setup, we've been able to implement a real-time research assistant, that also utilises the arXiv and Gemini APIS.
