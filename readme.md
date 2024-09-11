# Chat Analysis Project

## Project Goals

This project aims to analyze chat data from a text file to extract user interests and determine similarities between users. The main objectives are:

1. Parse chat data from a text file efficiently.
2. Extract user interests using natural language processing techniques.
3. Determine similarities between users based on their interests.
4. Provide insights into user engagement and topic preferences.

## Technical Details

### Technologies Used

- Python 3.8+
- spaCy
- scikit-learn
- Regular expressions (re module)

### Key Components

1. **Chat Parser**: Extracts structured data from the raw chat text file.
2. **Interest Extractor**: Uses NLP techniques to identify key topics and sentiments for each user.
3. **Similarity Calculator**: Computes similarities between users based on their interests.

### Algorithms and Models

- **Text Parsing**: Regular expressions for extracting date, time, user, and message content.
- **Interest Extraction**:
  - TF-IDF (Term Frequency-Inverse Document Frequency) for identifying important words.
  - spaCy's NLP pipeline for part-of-speech tagging and named entity recognition.
  - Simple rule-based approach for sentiment analysis.
- **Similarity Calculation**: Cosine similarity between user interest vectors.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/chat-analysis-project.git
   cd chat-analysis-project
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage

1. Prepare your chat data file in the required format (see `sample_chat.txt` for an example).

2. Run the analysis script:
   ```
   python analyze_chat.py path/to/your/chat_file.txt
   ```

3. View the results in the console output. (Future versions may include data visualization or export options.)

## Project Structure

```
chat-analysis-project/
│
├── analyze_chat.py        # Main script for running the analysis
├── chat_parser.py         # Module for parsing the chat file
├── interest_extractor.py  # Module for extracting user interests
├── similarity_calculator.py  # Module for calculating user similarities
├── requirements.txt       # List of Python package dependencies
├── README.md              # This file
└── sample_chat.txt        # Example chat file for testing
```

## Future Improvements

- Implement a more sophisticated sentiment analysis using spaCy's text classification capabilities.
- Fine-tune the spaCy model on domain-specific data for improved performance.
- Add topic modeling using techniques like LDA (Latent Dirichlet Allocation).
- Develop a user interface for easier interaction with the analysis tools.
- Incorporate time-based analysis to track interest changes over time.
- Add data visualization features for better insight presentation.
- Implement custom named entity recognition for domain-specific entities.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
