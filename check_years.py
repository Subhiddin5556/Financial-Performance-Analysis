import pandas as pd

years = [2014, 2015, 2016, 2017, 2018]

for year in years:
    df = pd.read_csv(f"{year}_Financial_Data.csv")

    targets = [
        col for col in df.columns
        if "PRICE VAR" in col.upper() or col == "Class"
    ]

    print(f"\n{year}:")
    print(targets)
# Check feature compatibility

feature_sets = {}

for year in years:
    df = pd.read_csv(f"{year}_Financ    ial_Data.csv")

    target = f"{year + 1} PRICE VAR [%]"

    features = set(
        df.drop(
            columns=["Class", target, "Unnamed: 0"],
            errors="ignore"
        ).columns
    )

    feature_sets[year] = features

for year in years:
    print(f"\n{year}: {len(feature_sets[year])} features")

print("\nDifferences between 2014 and 2018:")

print("Only in 2014:")
print(sorted(feature_sets[2014] - feature_sets[2018]))

print("\nOnly in 2018:")
print(sorted(feature_sets[2018] - feature_sets[2014]))