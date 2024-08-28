import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--filename", type=str, help="Input Filename")
    parser.add_argument("--features", type=str, help="Features name delimited by comma. e.g: feature1name,feature2name,feature3name")
    
    args = parser.parse_args()
    
    
    filename = args.filename
    features = args.features.split(",")
    
    
    df  = pd.read_csv(filename, names=features)
    
    df.to_csv(f"{filename.split('.')[0]}.csv", index=False)
    

if __name__ == "__main__":
    main()