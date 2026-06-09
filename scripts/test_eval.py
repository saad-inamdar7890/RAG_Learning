import requests
import sys
import json
import time

API_URL = "http://localhost:8000/api/ask"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

# A comprehensive test set representing our evaluation benchmark
TEST_CASES = [
    {"query": "What is covered under the standard fire policy?", "expected_topic": "fire damage coverage"},
    {"query": "Are floods covered by a standard homeowner's policy?", "expected_topic": "flood damage exclusions"},
    {"query": "How much liability coverage is included for personal injury?", "expected_topic": "personal liability limits"},
    {"query": "What is the standard deductible for collision coverage?", "expected_topic": "collision deductible"},
    {"query": "Is windstorm or hail damage covered?", "expected_topic": "wind and hail coverage"},
    {"query": "Do I need separate insurance for earthquakes?", "expected_topic": "earthquake coverage or exclusions"},
    {"query": "What is 'loss of use' coverage?", "expected_topic": "additional living expenses or loss of use"},
    {"query": "What are the limits for personal property coverage?", "expected_topic": "personal property limits"},
    {"query": "How is roof damage depreciation calculated?", "expected_topic": "roof depreciation or actual cash value"},
    {"query": "Does the policy include identity theft protection?", "expected_topic": "identity theft coverage"},
    {"query": "How can I insure expensive jewelry or art?", "expected_topic": "scheduled personal property or high-value items"},
    {"query": "Does this cover medical payments if someone gets hurt on my property?", "expected_topic": "medical payments to others"},
    {"query": "Are domestic workers covered under this policy?", "expected_topic": "workers compensation or domestic employees"},
    {"query": "What does an umbrella policy add?", "expected_topic": "umbrella limits or excess liability"},
    {"query": "Is vandalism covered if the home is vacant?", "expected_topic": "vandalism and vacancy clauses"},
    {"query": "What happens if I'm hit by an uninsured driver?", "expected_topic": "uninsured motorist coverage"},
    {"query": "Will the policy pay for a rental car while mine is being repaired?", "expected_topic": "rental reimbursement"},
    {"query": "Is glass breakage covered without a deductible?", "expected_topic": "glass breakage or zero deductible"},
    {"query": "Are there limitations on mold and fungus damage?", "expected_topic": "mold and fungus limits"},
    {"query": "Is damage from acts of terrorism covered?", "expected_topic": "terrorism coverage or TRIA"}
]

def check_backend():
    try:
        response = requests.get("http://localhost:8000/api/health")
        return response.status_code == 200
    except:
        return False

def call_rag_api(query: str):
    response = requests.post(API_URL, json={"query": query})
    if response.status_code != 200:
        raise RuntimeError(f"API Error: {response.text}")
    return response.json()

def llm_as_a_judge(query: str, answer: str, expected_topic: str) -> bool:
    """
    Uses the local LLM to evaluate if the answer adequately addresses the query
    based on the expected topic.
    """
    prompt = (
        "You are an impartial judge evaluating an AI assistant's answer.\n"
        f"Question: {query}\n"
        f"Assistant Answer: {answer}\n"
        f"Does the assistant's answer cover the topic of: {expected_topic}?\n"
        "Reply with exactly 'YES' or 'NO'."
    )
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    judgement = resp.json().get("response", "").strip().upper()
    
    return "YES" in judgement

def main():
    print("Starting CI Evaluation Pipeline...")
    
    # Wait for API to be ready
    for _ in range(60):
        if check_backend():
            break
        print("Waiting for RAG API backend...")
        time.sleep(2)
    else:
        print("ERROR: RAG API backend is not running at localhost:8000")
        sys.exit(1)

    passed = 0
    total = len(TEST_CASES)

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n--- Test {i}/{total}: '{test['query']}' ---")
        try:
            result = call_rag_api(test["query"])
            answer = result["answer"]
            sources = result["sources"]
            
            print(f"Retrieved {len(sources)} chunks.")
            
            is_good = llm_as_a_judge(test["query"], answer, test["expected_topic"])
            
            if is_good:
                print("[PASS] Answer correctly addresses the expected topic.")
                passed += 1
            else:
                print("[FAIL] Answer missed the expected topic.")
                print(f"Output was: {answer}")
                
        except Exception as e:
            print(f"[ERROR] during test: {e}")

    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Passed: {passed}/{total}")
    
    if passed < total:
        print("Status: FAILED")
        sys.exit(1)
    else:
        print("Status: PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()
