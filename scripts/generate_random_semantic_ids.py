import json
import random

def main():
    input_path = "./data/semantic_ids_steam.json"
    output_path = "./data/semantic_ids_random_steam.json"

    with open(input_path, "r") as f:
        original_data = json.load(f)

    random_data = {}
    for item_id, codes in original_data.items():
        # Generate random codes of the same length, assuming vocabulary size of 256 for each level
        random_codes = [random.randint(0, 255) for _ in codes]
        random_data[item_id] = random_codes

    with open(output_path, "w") as f:
        json.dump(random_data, f, indent=2)

    print(f"Generated {len(random_data)} random semantic IDs and saved to {output_path}")

if __name__ == "__main__":
    main()
