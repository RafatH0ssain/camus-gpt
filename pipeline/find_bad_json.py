import json

def check_file(filename):
    print(f"Checking {filename}...")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.decoder.JSONDecodeError as e:
                    print(f"\n❌ BROKEN JSON FOUND on line {i}:")
                    print(f"Error: {e}")
                    print(f"Line text: {line}")
                    return
        print("✅ No errors found in this file.")
    except FileNotFoundError:
        print(f"File {filename} not found.")

if __name__ == "__main__":
    # Check the extracted KB file (most likely culprit)
    check_file("./data/kb_extracted.jsonl")
    # Check the curated KB file just in case
    check_file("./camus_kb.jsonl")