# Chat Analysis Project

## Project Goals

This project aims to analyze chat data from a text file to extract user interests and determine similarities between users. The main objectives are:

1. Parse chat data from a text file efficiently.
2. Extract user interests using natural language processing techniques.
3. Determine similarities between users based on their interests.
4. Provide insights into user engagement and topic preferences.
5. Implement user identification based on linguistic features.
6. Analyze group chat dynamics and interest trends.
7. Predict potential user interests based on observed behavior.

## Technical Details

### Technologies Used

- Python 3.8+
- PyTorch
- Transformers library (Hugging Face)
- spaCy
- scikit-learn
- Regular expressions (re module)
- emoji library

### Key Components

1. **Chat Parser**: Extracts structured data from the raw chat text file.
2. **Interest Extractor**: Uses NLP techniques to identify key topics and sentiments for each user.
3. **Similarity Calculator**: Computes similarities between users based on their interests.
4. **User Identifier**: Utilizes linguistic features to distinguish between users.
5. **Group Dynamics Analyzer**: Examines interest trends and interactions within group chats.
6. **Interest Predictor**: Suggests potential interests based on user behavior and similarities.

### Algorithms and Models

- **Text Parsing**: Regular expressions for extracting date, time, user, and message content.
- **Interest Extraction**:
  - Transformer-based models (e.g., RoBERTa, BERT, XLNet) for feature extraction and classification.
  - TF-IDF (Term Frequency-Inverse Document Frequency) for identifying important words.
  - spaCy's NLP pipeline for part-of-speech tagging and named entity recognition.
  - Latent Dirichlet Allocation (LDA) for topic modeling.
  - Fine-tuned sentiment analysis using DistilBERT or similar models.
- **Similarity Calculation**: Cosine similarity between user interest vectors or embeddings.
- **User Identification**: Combination of linguistic features and transformer model outputs.
- **Interest Prediction**: Classifier trained on transformer model outputs and user behavior data.

## Recommended Models

We recommend using the following transformer-based models for your chat analysis:

1. **RoBERTa-base or RoBERTa-large**
   - Good for general language understanding and feature extraction
   - Can be fine-tuned for specific tasks like interest identification

2. **BERT-base-multilingual-cased**
   - Useful if your chat includes multiple languages or code-switching

3. **XLNet-base-cased**
   - Known for capturing long-range dependencies, which could be useful for understanding context in chat conversations

4. **DistilBERT-base-uncased-finetuned-sst-2-english**
   - Specifically fine-tuned for sentiment analysis, which could be useful for scoring user interests

5. **ALBERT-base-v2**
   - More efficient than BERT, good for handling longer sequences of text

These models can be used interchangeably or in combination, depending on the specific requirements of your chat analysis task. The choice of model may depend on factors such as the language(s) used in the chat, the length of messages, and the specific analysis goals (e.g., sentiment analysis, interest identification).

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
   python -m spacy download en_core_web_trf
   ```

## Usage

### Chat Parser

1. Prepare your chat data file in the required format (see `sample_chat.txt` for an example).

2. Run the chat parser script:
   ```
   python chat_parser.py path/to/your/chat_file.txt
   ```

3. View the results in the console output. The script will provide the following statistics:
   - Total number of messages
   - Messages per user
   - Date range of the chat
   - Top 5 most active users
   - Average time between messages for each user
   - Most common words used in the chat

4. The parsed data will also be saved to a JSON file named `parsed_chat_data.json` in the same directory.

### Full Analysis

1. Run the analysis script:
   ```
   python analyze_chat.py path/to/your/chat_file.txt
   ```

2. View the results in the console output. The analysis will include:
   - Basic chat statistics
   - Identified user interests
   - User similarity scores
   - Group chat dynamics analysis
   - Predicted potential interests for users

3. (Optional) Generate visualizations of the analysis results using the included plotting functions.

## Project Structure

```
chat-analysis-project/
│
├── analyze_chat.py        # Main script for running the full analysis
├── chat_parser.py         # Module for parsing the chat file and basic statistics
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
