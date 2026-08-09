import subprocess
import sys

PROMPTS = [
    # --- PETS & CAT (Identity Card) ---
    "do you have a cat",
    "do you have a pet",
    "whats your cats name",
    "your cat's name?",
    "tell me about your pets",
    "did you have a dog",
    "what animals do you keep?",
    "what were your dogs' names?",
    "did you ever own a pet?",
    "was your cat named Mersault?",

    # --- WORKS & CATEGORIES (Identity Card) ---
        "name your works",
    "list your novels",
    "what plays did you write?",
    "tell me about your essays",
    "is The Rebel a novel?",
    "what short stories did you publish?",
    "can you list all your books?",
    "did you write The Stranger?",
    "name your posthumous works",
    "bibliography please",

    # --- BIOGRAPHY & QUOTES (Identity Card) ---
    "are you an existentialist?",
    "why did you fall out with Sartre?",
    "did you say 'don't walk behind me'?",
    "tell me about the invincible summer",
    "where were you born?",
    "who was Louis Germain?",
    "did you play any sports?",
    "when did you win the Nobel prize?",
    "what happened in 1952?",
    "how did you die?"

]

def run_tests():
    print(f"Starting automated test battery: {len(PROMPTS)} prompts...\n")
    
    # Join all prompts with newlines. When stdin runs out, camus_rag.py 
    # will hit an EOFError and exit cleanly.
    input_data = "\n".join(PROMPTS) + "\n"
    
    try:
        # Pass the prompts directly to stdin.
        # sys.executable ensures it uses your active (.venv) Python.
        subprocess.run(
            [sys.executable, "camus_rag.py", "--debug"],
            input=input_data,
            text=True,
            check=True
        )
    except KeyboardInterrupt:
        print("\nTest battery aborted by user.")
    except Exception as e:
        print(f"\nError running tests: {e}")
        
    print("\n=== TEST BATTERY COMPLETE ===")

if __name__ == "__main__":
    run_tests()