import json
import matplotlib.pyplot as plt

def main():
    with open('data/processed/baseline_results.json', 'r') as f:
        baseline = json.load(f)

    with open('data/processed/advanced_results.json', 'r') as f:
        advanced = json.load(f)
    
    metrics = ['accuracy', 'bleu', 'word_accuracy', 'char_accuracy', 'noun_accuracy']
    
    baseline_values = [baseline[f'baseline_{m}'] for m in metrics]
    advanced_values = [advanced[f'advanced_{m}'] for m in metrics]
    
    x = range(len(metrics))
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, baseline_values, width=0.4, label='Baseline', align='center')
    plt.bar([i + 0.4 for i in x], advanced_values, width=0.4, label='Advanced', align='center')
    plt.xticks([i + 0.2 for i in x], metrics)
    plt.ylabel('Score')
    plt.title('Baseline vs Advanced Model Performance Comparison')
    plt.legend()
    plt.savefig('plots/comparison.png')
    plt.close()
    
    print("Comparison plot saved to plots/comparison.png")
    print("Baseline:", baseline)
    print("Advanced:", advanced)

if __name__ == "__main__":
    main()