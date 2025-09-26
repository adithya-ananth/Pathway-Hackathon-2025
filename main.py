import subprocess
import sys

scripts = [
    "keyword_extractor.py",
    "scraper.py",
    "parser.py",
    "enrich_papers.py",
    "rag/main.py"
]

def run_scripts():
    for script in scripts:
        print(f"\nRunning {script} ...")
        try:
            # Run the script with the same Python interpreter
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
    print("\nAll scripts completed successfully!")

if __name__ == "__main__":
    run_scripts()