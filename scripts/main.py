import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

def load_df(data_path : str, extension_pattern : str) -> pd.DataFrame:
    data_dir = Path.cwd() / data_path 
    data_file_extsn = f"*.{extension_pattern}" 
    data_file_path = list(data_dir.glob(data_file_extsn))[0]
    df = pd.read_excel(data_file_path)
    return df

def get_sentence_length(sen : str) -> int :
    return len(list(sen.split(" ")))

def main():

    # Load xlsx into pandas dataframe
    df = load_df("data", "xlsx")

    # Swap columns: target | features ->  features | target
    df = df[[df.columns[1], df.columns[0]]]

    # -> 4
    print(get_sentence_length("Hi how are you"))
    return print(df[df.columns[0]].head())

if __name__ == "__main__":
    main()
