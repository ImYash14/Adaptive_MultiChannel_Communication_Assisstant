# Cortex: Adaptive Multi Channel Communication Assistant

A sophisticated customer feedback analysis system built with Python, leveraging AI to process, analyze, and respond to customer feedback automatically.

## Features

- Sentiment analysis of customer feedback
- Automatic response generation based on feedback content
- FAQ integration and similarity search
- Support for multiple product categories
- Web interface for easy interaction
- Detailed aspect-based sentiment breakdown
- Real-time feedback processing

## Technologies Used

- Flask web framework
- Groq LLM API for natural language processing
- Langchain for LLM integration
- HuggingFace embeddings for semantic search
- ChromaDB for vector storage
- Pydantic for data validation
- Bootstrap for frontend styling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yaryamantiwari17/just-another-friday.git
cd just-another-friday
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add:
```
GROQ_API_KEY=your_groq_api_key
```

## Usage

1. Start the Flask application:
```bash
python app.py or for terminal use running.py
```

2. Navigate to `http://localhost:5000` in your web browser or just enter the basic details through terminal.

3. Initialize the system:
   - Enter your company name
   - Choose FAQ input method (direct input or JSON file)
   - Enter product categories
   - Click "Initialize System"

4. Start analyzing feedback:
   - Enter customer feedback in the text area
   - Click "Analyze Feedback"
   - View detailed analysis results and generated responses

## Project Structure

```
├── app.py                 # Flask application entry point
├── utils/
│   └── feedback_utils.py  # Core functionality
├── templates/
│   ├── base.html         # Base template
│   ├── index.html        # System initialization
│   ├── feedback.html     # Feedback input
│   └── results.html      # Analysis results
├── static/
│   └── styles.css        # Custom styles
└── requirements.txt      # Project dependencies
```
## Video Demonstration
https://github.com/user-attachments/assets/1f9d30e1-1ecf-4d63-add2-cc37aef6f573

## Features in Detail

### Sentiment Analysis
- Overall sentiment detection
- Aspect-based sentiment breakdown
- Mixed sentiment detection
- Score-based sentiment intensity

### Response Generation
- Context-aware responses
- Professional and empathetic tone
- Solution-focused suggestions
- Integration with FAQ knowledge base

### FAQ Integration
- Semantic similarity search
- Relevant FAQ matching
- Support for direct input and JSON file loading

## Output
![1](https://github.com/user-attachments/assets/6de0e671-1323-4c1d-8984-654d4767c79e)
![2](https://github.com/user-attachments/assets/0ce086c2-1f6e-4969-90bd-ed0a3df8a754)
![3](https://github.com/user-attachments/assets/07d9251a-981b-4dd9-aaa0-2ceec82c5229)
![4](https://github.com/user-attachments/assets/148afcae-73fb-4dc5-a805-74cfa53c72d5)
![result](https://github.com/user-attachments/assets/d3b4fbbf-cd18-417d-b8a6-42dce0fb9e2b)


## Acknowledgments

- Built with Groq's LLM API
- Powered by Langchain framework
- Uses HuggingFace's powerful embeddings
- ChromaDB for efficient vector storage
