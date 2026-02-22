from pathlib import Path

def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    print("OK: markov_trnasitions (placeholder)")

if __name__ == "__main__":
    main()