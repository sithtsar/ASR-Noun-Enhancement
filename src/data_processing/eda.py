import pandas as pd
from pathlib import Path
import json
from collections import Counter
import re
import spacy
import nltk
from nltk import pos_tag, word_tokenize

def load_df(data_path: str, extension_pattern: str) -> pd.DataFrame:
    data_dir = Path.cwd() / data_path
    data_file_extsn = f"*.{extension_pattern}"
    data_file_path = list(data_dir.glob(data_file_extsn))[0]
    df = pd.read_excel(data_file_path)
    return df

def get_sentence_length(sen: str) -> int:
    return len(sen.split())

def clean_text(text: str) -> str:
    """Clean text: lowercase, remove extra spaces, handle punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # normalize spaces
    return text

def load_spacy_model():
    """Load spacy model, download if needed."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading en_core_web_sm...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

def extract_nouns_ner(text: str, nlp) -> list:
    """Extract nouns using NER."""
    doc = nlp(text)
    nouns = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]  # focus on entities that could be medical
    return nouns

def extract_nouns_pos(text: str) -> list:
    """Extract nouns using POS tagging."""
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [word for word, tag in tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
    return nouns

def compute_stats(df):
    # Clean text
    df['clean_correct'] = df['correct sentences'].apply(clean_text)
    df['clean_incorrect'] = df['ASR-generated incorrect transcriptions'].apply(clean_text)

    # Sentence lengths
    df['len_correct'] = df['clean_correct'].apply(get_sentence_length)
    df['len_incorrect'] = df['clean_incorrect'].apply(get_sentence_length)

    # Load models
    nlp = load_spacy_model()

    # Extract nouns
    df['nouns_correct_ner'] = df['correct sentences'].apply(lambda x: extract_nouns_ner(x, nlp))
    df['nouns_incorrect_ner'] = df['ASR-generated incorrect transcriptions'].apply(lambda x: extract_nouns_ner(x, nlp))
    df['nouns_correct_pos'] = df['correct sentences'].apply(extract_nouns_pos)
    df['nouns_incorrect_pos'] = df['ASR-generated incorrect transcriptions'].apply(extract_nouns_pos)

    # Vocabulary
    correct_words = ' '.join(df['clean_correct']).split()
    incorrect_words = ' '.join(df['clean_incorrect']).split()

    vocab_correct = set(correct_words)
    vocab_incorrect = set(incorrect_words)

    # Medical terms: capitalized words (simple proxy)
    medical_correct = [w for w in correct_words if w[0].isupper()]
    medical_incorrect = [w for w in incorrect_words if w[0].isupper()]
    
    # Error stats: simple diff count
    def count_diffs(row):
        correct = set(row['correct sentences'].split())
        incorrect = set(row['ASR-generated incorrect transcriptions'].split())
        return len(correct.symmetric_difference(incorrect))
    
    df['num_errors'] = df.apply(count_diffs, axis=1)
    
    stats = {
        'total_samples': len(df),
        'avg_len_correct': df['len_correct'].mean(),
        'avg_len_incorrect': df['len_incorrect'].mean(),
        'vocab_size_correct': len(vocab_correct),
        'vocab_size_incorrect': len(vocab_incorrect),
        'medical_terms_correct': len(set(medical_correct)),
        'medical_terms_incorrect': len(set(medical_incorrect)),
        'avg_errors_per_sentence': df['num_errors'].mean(),
        'avg_nouns_correct_ner': df['nouns_correct_ner'].apply(len).mean(),
        'avg_nouns_incorrect_ner': df['nouns_incorrect_ner'].apply(len).mean(),
        'avg_nouns_correct_pos': df['nouns_correct_pos'].apply(len).mean(),
        'avg_nouns_incorrect_pos': df['nouns_incorrect_pos'].apply(len).mean()
    }
    
    return df, stats

def main():
    df = load_df("data", "xlsx")
    augmented_df, stats = compute_stats(df)
    
    # Data split: 70-15-15
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(augmented_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Save splits
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)

    # Save full augmented df
    augmented_df.to_csv('data/processed/augmented_df.csv', index=False)

    print("Preprocessing and EDA completed. Augmented DF, splits, and stats saved to data/processed/")

    # Save stats
    with open('data/processed/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("EDA completed. Augmented DF, splits, and stats saved to data/processed/")

if __name__ == "__main__":
    main()