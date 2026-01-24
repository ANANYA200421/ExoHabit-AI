import pandas as pd

# Load your Excel file
df = pd.read_excel('More data added.xlsx')  # Replace with your actual filename

print("=" * 80)
print("DIAGNOSTIC CHECK")
print("=" * 80)

print(f"\nTotal rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\n📋 ALL COLUMN NAMES:")
print("-" * 80)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. '{col}'")

print("\n" + "=" * 80)
print("Copy these column names and send them to me!")
print("=" * 80)

# Also check first few rows
print("\nFirst 3 rows preview:")
print(df.head(3))