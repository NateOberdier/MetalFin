import pandas as pd
import numpy as np
import random

random.seed(50)
np.random.seed(50)

HG_RANGES = {
    "very_low":  (0.001, 0.05),
    "low":       (0.02, 0.10),
    "medium":    (0.10, 0.30),
    "high":      (0.30, 0.70),
    "very_high": (0.70, 1.50),
}

LEAD_RANGES = {
    "very_low":  (0.0001, 0.003),
    "low":       (0.003, 0.010),
    "medium":    (0.010, 0.030),
    "high":      (0.030, 0.100),
}

ARSENIC_RANGES = {
    "very_low":  (0.005,  0.050),
    "low":       (0.050,  0.200),
    "medium":    (0.200,  0.500),
    "high":      (0.500,  1.500),
}

CADMIUM_RANGES = {
    "very_low":  (0.00001, 0.00020),
    "low":       (0.00020, 0.00100),
    "medium":    (0.00100, 0.00500),
    "high":      (0.00500, 0.05000),
}

def draw_from_range(ranges_dict, tier):
    lo, hi = ranges_dict[tier]
    return np.random.uniform(lo, hi)

species_data = [
    ("Atlantic Salmon", "marine", "carnivore", (60, 90),   (3000, 6000), "very_low", "very_low", "low",      "very_low"),
    ("Coho Salmon",     "marine", "carnivore", (55, 80),   (2500, 5000), "very_low", "very_low", "low",      "very_low"),
    ("Sockeye Salmon",  "marine", "carnivore", (50, 75),   (2000, 4000), "very_low", "very_low", "low",      "very_low"),
    ("Salmon (Generic)","marine", "carnivore", (50, 90),   (2000, 6000), "very_low", "very_low", "low",      "very_low"),
    ("Shrimp",          "marine", "omnivore",  (8, 20),    (10, 40),     "very_low", "low",      "medium",   "low"),
    ("Pollock",         "marine", "carnivore", (40, 90),   (1000, 5000), "very_low", "very_low", "low",      "very_low"),
    ("Tilapia",         "freshwater", "herbivore", (20, 45), (500, 1500),"very_low", "low",      "very_low", "very_low"),
    ("Catfish",         "freshwater", "omnivore", (25, 60), (800, 3000), "very_low", "low",      "very_low", "very_low"),
    ("Cod",             "marine", "carnivore", (50, 100),  (2000, 7000), "low",      "low",      "low",      "very_low"),
    ("Haddock",         "marine", "carnivore", (40, 75),   (1000, 3000), "low",      "low",      "low",      "very_low"),
    ("Whiting",         "marine", "carnivore", (30, 60),   (800, 2500),  "low",      "low",      "low",      "very_low"),
    ("Sardine",         "marine", "carnivore", (12, 25),   (50, 150),    "very_low", "low",      "medium",   "very_low"),
    ("Anchovy",         "marine", "carnivore", (8, 15),    (20, 50),     "very_low", "low",      "medium",   "very_low"),
    ("Herring",         "marine", "carnivore", (20, 35),   (100, 400),   "low",      "low",      "medium",   "very_low"),
    ("Mahi-Mahi",       "marine", "carnivore", (70, 120),  (4000, 15000),"medium",   "medium",   "low",      "low"),
    ("Albacore Tuna",   "marine", "carnivore", (80, 140),  (8000, 40000),"medium",   "medium",   "low",      "low"),
    ("Skipjack Tuna",   "marine", "carnivore", (60, 100),  (5000, 25000),"medium",   "medium",   "low",      "low"),
    ("Bluefish",        "marine", "carnivore", (40, 80),   (2000, 6000), "medium",   "medium",   "low",      "low"),
    ("Sea Bass",        "marine", "carnivore", (30, 80),   (1000, 7000), "medium",   "medium",   "low",      "low"),
    ("Snapper",         "marine", "carnivore", (30, 70),   (700, 3500),  "medium",   "medium",   "low",      "low"),
    ("Grouper",         "marine", "carnivore", (40, 100),  (1500, 10000),"high",     "medium",   "low",      "low"),
    ("Halibut",         "marine", "carnivore", (80, 200),  (20000, 60000),"medium",  "medium",   "low",      "low"),
    ("Trout",           "freshwater", "carnivore", (25, 60), (300, 2000),"low",      "low",      "very_low", "very_low"),
    ("Perch",           "freshwater", "carnivore", (15, 40), (200, 800), "low",      "low",      "very_low", "very_low"),
    ("Carp",            "freshwater", "omnivore",  (30, 70), (1000, 6000),"medium",  "medium",   "low",      "low"),
    ("Walleye",         "freshwater", "carnivore", (30, 80), (800, 4000),"medium",   "medium",   "low",      "low"),
    ("Pike",            "freshwater", "carnivore", (40, 120),(1500, 10000),"medium", "medium",   "low",      "low"),
    ("Bass (Freshwater)", "freshwater","carnivore",(25, 60),(500, 3500), "medium",   "medium",   "low",      "low"),
    ("Swordfish",       "marine", "carnivore", (100, 250), (20000, 50000),"very_high","medium","low","low"),
    ("Shark",           "marine", "carnivore", (150, 400), (30000, 200000),"very_high","medium","low","low"),
    ("King Mackerel",   "marine", "carnivore", (50, 120), (2000, 15000),"very_high", "medium","low","low"),
    ("Tilefish (Gulf)", "marine", "carnivore", (40, 80),  (1500, 7000), "very_high", "medium","low","low"),
    ("Orange Roughy",   "marine", "carnivore", (40, 75),  (1000, 5000), "high",      "medium","low","low"),
    ("Marlin",          "marine", "carnivore", (150, 300),(30000, 200000),"very_high","medium","low","low"),
    ("Bluefin Tuna",    "marine", "carnivore", (150, 250),(30000, 250000),"very_high","medium","low","low"),
    ("Bigeye Tuna",     "marine", "carnivore", (100, 200),(15000, 70000),"high",     "medium","low","low"),
    ("Yellowfin Tuna",  "marine", "carnivore", (100, 200),(15000, 70000),"high",     "medium","low","low"),
    ("Escolar",         "marine", "carnivore", (80, 200), (8000, 60000),"high",      "medium","low","low"),
    ("Opah",            "marine", "carnivore", (80, 180), (10000, 80000),"high",     "medium","low","low"),
    ("Flounder",        "marine", "carnivore", (30, 60),  (600, 2000), "low",       "low",   "low",      "very_low"),
    ("Plaice",          "marine", "carnivore", (25, 55),  (500, 1800), "low",       "low",   "low",      "very_low"),
    ("Sole",            "marine", "carnivore", (25, 60),  (600, 2000), "low",       "low",   "low",      "very_low"),
    ("Lobster (American)", "marine", "omnivore",(20, 50), (400, 2000), "low",      "medium","medium","medium"),
    ("Lobster (Spiny)",    "marine", "omnivore",(20, 50), (400, 2000), "low",      "medium","medium","medium"),
    ("Crab",               "marine", "omnivore",(10, 25), (150, 800),  "low",      "medium","medium","medium"),
    ("Crayfish",           "freshwater","omnivore",(8, 15),(20, 100),  "very_low", "low",   "medium","low"),
    ("Oyster",          "marine", "filter_feeder",(6, 15), (20, 100),  "low",      "medium","high",   "medium"),
    ("Mussel",          "marine", "filter_feeder",(5, 12), (15, 80),   "low",      "medium","high",   "medium"),
    ("Clam",            "marine", "filter_feeder",(5, 10), (10, 60),   "low",      "medium","high",   "medium"),
    ("Scallop",         "marine", "filter_feeder",(5, 12), (15, 70),   "low",      "medium","high",   "medium"),
    ("Squid",           "marine", "carnivore",   (10, 40), (50, 500),  "low",      "low",   "medium",  "high"),
    ("Octopus",         "marine", "carnivore",   (20, 80), (200, 4000),"low",      "low",   "medium",  "high"),
    ("Bluegill",        "freshwater", "omnivore",(15, 30), (50, 300),  "low",      "low",   "very_low","very_low"),
    ("Whitefish",       "freshwater", "carnivore",(30, 60),(800, 3000),"low",      "low",   "low",     "very_low"),
]

locations = [
    "North Atlantic", "South Atlantic", "Pacific Ocean", "Indian Ocean",
    "Mediterranean Sea", "Bering Sea", "Gulf of Mexico", "Caribbean Sea",
    "Arctic Ocean", "North Sea", "Baltic Sea",
    "Mississippi River", "Great Lakes", "Amazon River",
    "Bay of Biscay", "Sea of Japan", "Gulf of Thailand"
]

def generate_synthetic_fish_dataset(n_rows=500, missing_prob=0.0):
    rows = []

    for _ in range(n_rows):
        (species, habitat, diet, length_range, weight_range,
         hg_tier, pb_tier, as_tier, cd_tier) = random.choice(species_data)

        location = random.choice(locations)
        length = np.random.uniform(*length_range)
        weight = np.random.uniform(*weight_range)

        mercury = draw_from_range(HG_RANGES, hg_tier)
        lead    = draw_from_range(LEAD_RANGES, pb_tier)
        arsenic = draw_from_range(ARSENIC_RANGES, as_tier)
        cadmium = draw_from_range(CADMIUM_RANGES, cd_tier)

        if np.random.rand() < missing_prob:
            mercury = np.nan
        if np.random.rand() < missing_prob:
            lead = np.nan
        if np.random.rand() < missing_prob:
            arsenic = np.nan
        if np.random.rand() < missing_prob:
            cadmium = np.nan

        rows.append({
            "species": species,
            "location": location,
            "habitat": habitat,
            "diet": diet,
            "length_cm": round(length, 2),
            "weight_g": round(weight, 2),
            "mercury_mg_kg": round(mercury, 4) if not np.isnan(mercury) else np.nan,
            "lead_mg_kg":     round(lead,    4) if not np.isnan(lead)    else np.nan,
            "arsenic_mg_kg":  round(arsenic, 4) if not np.isnan(arsenic) else np.nan,
            "cadmium_mg_kg":  round(cadmium, 4) if not np.isnan(cadmium) else np.nan,
        })

    return pd.DataFrame(rows)

df_large = generate_synthetic_fish_dataset(n_rows=1000, missing_prob=0.15)

df_large.to_csv("fish_heavy_metals_master.csv", index=False)

print(f"✔ Generated synthetic dataset with {len(df_large)} rows:")
display(df_large.head())
print(df_large["species"].value_counts().head(15))
