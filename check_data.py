from datasets import load_from_disk
import os

# Load dataset
ds = load_from_disk(os.path.join('data', 'local_diabetes_dataset'))

print(f"Total records: {len(ds)}")
print("\nFirst 10 questions:")
for i in range(min(10, len(ds))):
    print(f"{i+1}. {ds[i]['question']}")

print("\n" + "="*60)
print("Checking for 'symptoms' questions:")
symptoms_count = 0
for i in range(len(ds)):
    q = ds[i]['question'].lower()
    if 'symptom' in q:
        symptoms_count += 1
        if symptoms_count <= 5:
            print(f"- {ds[i]['question']}")

print(f"\nTotal questions about symptoms: {symptoms_count}")
