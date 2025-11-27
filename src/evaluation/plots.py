import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import Counter
import ast

def plot_sentence_lengths(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['len_correct'], alpha=0.5, label='Correct', bins=20)
    plt.hist(df['len_incorrect'], alpha=0.5, label='Incorrect', bins=20)
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentence Lengths')
    plt.legend()
    plt.savefig('plots/sentence_lengths.png')
    plt.close()

def plot_error_categories(df):
    counters = df['error_categories'].dropna().apply(ast.literal_eval)
    total = Counter()
    for c in counters:
        total.update(c)
    plt.figure(figsize=(10, 6))
    plt.bar(list(total.keys()), list(total.values()))
    plt.xlabel('Error Type')
    plt.ylabel('Frequency')
    plt.title('Frequency of Error Types')
    plt.savefig('plots/error_types.png')
    plt.close()

def plot_error_distribution(df):
    # For simplicity, plot histogram of num_errors
    plt.figure(figsize=(10, 6))
    plt.hist(df['num_errors'], bins=range(0, df['num_errors'].max()+2), alpha=0.7)
    plt.xlabel('Number of Errors per Sentence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Errors per Sentence')
    plt.savefig('plots/error_distribution.png')
    plt.close()

def plot_vocab_stats():
    with open('data/processed/stats.json', 'r') as f:
        stats = json.load(f)

    labels = ['Correct Vocab', 'Incorrect Vocab', 'Medical Correct', 'Medical Incorrect']
    sizes = [stats['vocab_size_correct'], stats['vocab_size_incorrect'], stats['medical_terms_correct'], stats['medical_terms_incorrect']]

    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, explode=(0.05, 0.05, 0.05, 0.05))
    plt.title('Vocabulary Coverage Statistics', fontsize=14, fontweight='bold')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.savefig('plots/vocab_stats.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_baseline_results():
    with open('data/processed/baseline_results.json', 'r') as f:
        results = json.load(f)
    
    metrics = ['Accuracy', 'BLEU']
    values = [results['baseline_accuracy'], results['baseline_bleu']]
    
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'green'])
    plt.ylabel('Score')
    plt.title('Baseline Model Performance')
    plt.ylim(0, 1)
    plt.savefig('plots/baseline_performance.png')
    plt.close()

def plot_length_impact():
    df = pd.read_csv('data/processed/augmented_df.csv')
    plt.figure(figsize=(10, 6))
    plt.scatter(df['len_correct'], df['num_errors'])
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Number of Errors')
    plt.title('Context Length Impact on Correction Errors')
    plt.savefig('plots/length_impact.png')
    plt.close()

def plot_noun_error_distribution():
    df = pd.read_csv('data/processed/augmented_df.csv')
    # Simple: plot avg nouns vs errors
    plt.figure(figsize=(10, 6))
    plt.scatter(df['nouns_correct_pos'].apply(len), df['num_errors'])
    plt.xlabel('Number of Nouns in Sentence')
    plt.ylabel('Number of Errors')
    plt.title('Noun Count vs Errors')
    plt.savefig('plots/noun_error_dist.png')
    plt.close()

def main():
    df = pd.read_csv('data/processed/augmented_df.csv')
    plot_sentence_lengths(df)
    plot_error_categories(df)
    plot_error_distribution(df)
    plot_vocab_stats()
    plot_baseline_results()
    plot_length_impact()
    plot_noun_error_distribution()
    print("Plots generated and saved in plots/")

if __name__ == "__main__":
    main()