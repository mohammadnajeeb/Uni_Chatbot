# Faiz Chatbot

A chatbot trained on data from AMU (Aligarh Muslim University) websites, designed to answer questions in a professional yet engaging manner.

## Project Structure

- `scrapers/`: Web scraping scripts for collecting data from AMU websites
- `data/`: Storage for scraped and processed data
- `chatbot/`: Core chatbot implementation and model training
- `utils/`: Utility functions and helpers
- `frontend/`: User interface components

## Setup and Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys (see `.env.example`)
4. Run the scraper:
   ```
   python scrapers/main_scraper.py
   ```
5. Launch the chatbot:
   ```
   python main.py
   ```

## Features

- Data scraping from www.amu.ac.in and amucontrollerexams.com
- Data processing pipeline for text extraction and cleaning
- Vector database for efficient information retrieval
- Interactive web interface for chatting
- Professional yet fun conversation style

## License

MIT