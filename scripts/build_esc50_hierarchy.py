"""Build data/esc50_bga_labels.csv: ESC-50 category -> mid-category -> BGA
(biophony / geophony / anthrophony, Pijanowski et al. 2011)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pandas as pd

from hierbio import paths

META_PATH = paths.ESC50_META
OUT_PATH = paths.ESC50_LABELS

df = pd.read_csv(META_PATH)

# Levels: BGA (coarse) -> mid-category -> the original 50 categories (fine)

midcat_map = {
    # Biophony
    "dog": "mammal", "pig": "mammal", "cow": "mammal", "cat": "mammal", "sheep": "mammal",
    "rooster": "bird", "hen": "bird", "crow": "bird", "chirping_birds": "bird",
    "insects": "insect", "crickets": "insect",
    "frog": "amphibian",
    # Geophony
    "rain": "water", "sea_waves": "water", "water_drops": "water", "pouring_water": "water",
    "wind": "air", "thunderstorm": "air",
    "crackling_fire": "fire",
    # Anthrophony
    "crying_baby": "human_body", "sneezing": "human_body", "clapping": "human_body",
    "breathing": "human_body", "coughing": "human_body", "footsteps": "human_body",
    "laughing": "human_body", "brushing_teeth": "human_body", "snoring": "human_body",
    "drinking_sipping": "human_body",
    "toilet_flush": "domestic", "door_wood_knock": "domestic", "mouse_click": "domestic",
    "keyboard_typing": "domestic", "door_wood_creaks": "domestic", "can_opening": "domestic",
    "washing_machine": "domestic", "vacuum_cleaner": "domestic", "clock_alarm": "domestic",
    "clock_tick": "domestic", "glass_breaking": "domestic",
    "helicopter": "urban_transport", "car_horn": "urban_transport", "engine": "urban_transport",
    "train": "urban_transport", "airplane": "urban_transport",
    "chainsaw": "tools", "hand_saw": "tools",
    "siren": "signal_alarm", "church_bells": "signal_alarm", "fireworks": "signal_alarm",
}

bga_map = {
    "mammal": "biophony", "bird": "biophony", "insect": "biophony", "amphibian": "biophony",
    "water": "geophony", "air": "geophony", "fire": "geophony",
    "human_body": "anthrophony", "domestic": "anthrophony", "urban_transport": "anthrophony",
    "tools": "anthrophony", "signal_alarm": "anthrophony",
}


unmapped = set(df["category"].unique()) - set(midcat_map.keys())
assert not unmapped, f"unmapped categories: {unmapped}"

df["midcat"]    = df["category"].map(midcat_map)
df["bga_label"] = df["midcat"].map(bga_map)

print(df["bga_label"].value_counts())
print()
print(df["midcat"].value_counts())
print("\nTotal:", len(df))
print("Unique categories covered:", df["category"].nunique(), "/ 50")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")