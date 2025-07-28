# Fake News Generator & Detector using Generative AI and NLP

This project includes:
- A Fake News Detector using NLP and machine learning (PassiveAggressiveClassifier + TF-IDF)
- A Fake News Generator using a pre-trained GPT-2 model via HuggingFace Transformers
- Streamlit-based UI

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

Place your dataset (`fake_or_real_news.csv`) in the `data/` folder.
