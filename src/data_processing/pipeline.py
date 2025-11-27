import pandas as pd
import json
from difflib import SequenceMatcher
from collections import defaultdict, Counter
import re
import spacy
import spacy

def get_differing_words(correct, incorrect):
    """Extract pairs of differing words between correct and incorrect sentences."""
    correct_words = correct.split()
    incorrect_words = incorrect.split()
    
    matcher = SequenceMatcher(None, correct_words, incorrect_words)
    blocks = matcher.get_matching_blocks()
    
    errors = []
    i = 0
    for block in blocks:
        # Words before the match in correct
        for j in range(i, block.a):
            if block.b > 0:
                errors.append((correct_words[j], incorrect_words[block.b - (block.a - j)]))
            else:
                errors.append((correct_words[j], ''))
        # Words before the match in incorrect
        for j in range(i, block.b):
            if block.a > 0:
                pass  # already handled
        i = block.a + block.size
    
    # Remaining words
    for j in range(i, len(correct_words)):
        errors.append((correct_words[j], ''))
    for j in range(i, len(incorrect_words)):
        errors.append(('', incorrect_words[j]))
    
    return errors

def categorize_error(incorrect, correct):
    """Categorize the error type."""
    if not incorrect or not correct:
        return 'segmentation'  # missing or extra words

    if incorrect.lower() == correct.lower():
        return 'case'  # case difference

    # Edit distance
    matcher = SequenceMatcher(None, incorrect, correct)
    dist = len(incorrect) + len(correct) - 2 * sum(block.size for block in matcher.get_matching_blocks())

    if dist == 0:
        return 'none'
    elif dist <= 2:
        return 'character'  # small edits
    elif len(incorrect.split()) != len(correct.split()):
        return 'segmentation'  # word boundaries
    else:
        # Check if phonetic: similar length, high similarity ratio
        ratio = matcher.ratio()
        if ratio > 0.6:  # threshold for phonetic
            return 'phonetic'
        else:
            return 'word'

def build_error_db(errors_list):
    """Build a database of error patterns: incorrect -> list of possible corrects."""
    db = defaultdict(list)
    for errors in errors_list:
        for cor, inc in errors:  # Note: errors is (correct, incorrect), so cor is correct, inc is incorrect
            if inc and cor:
                db[inc].append(cor)
    
    # Keep most common correction for each
    final_db = {}
    for inc, cors in db.items():
        most_common = Counter(cors).most_common(1)[0][0]
        final_db[inc] = most_common
    
    return final_db

def main():
    # Load train df
    df = pd.read_csv('data/processed/train.csv')
    
    # Extract errors
    df['error_words'] = df.apply(lambda row: get_differing_words(row['correct sentences'], row['ASR-generated incorrect transcriptions']), axis=1)
    
    # Categorize
    def categorize_row(errors):
        cats = [categorize_error(inc, cor) for inc, cor in errors if inc and cor]
        return Counter(cats)
    
    df['error_categories'] = df['error_words'].apply(lambda x: dict(categorize_row(x)))
    
    # Build error DB
    all_errors = df['error_words'].tolist()
    error_db = build_error_db(all_errors)
    
    # Save updated df
    df.to_csv('data/processed/augmented_df.csv', index=False)

    # Save error DB
    with open('data/processed/error_db_train.json', 'w') as f:
        json.dump(error_db, f, indent=2)
    
    print("Pipeline completed. Error DB built from train data.")

if __name__ == "__main__":
    main()