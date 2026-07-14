import pandas as pd
from pathlib import Path

BASE_DIR  = Path("/user/gaad2403/bachelor_thesis")
META_PATH = BASE_DIR / "ESC-50-master/meta/esc50.csv"
OUT_PATH  = Path("/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv")

df = pd.read_csv(META_PATH)

# Level 1 (= "family" 자리): BGA
# Level 2 (= "genus" 자리): 중분류
# Level 3 (= "species" 자리): category (원본 50개, 그대로 사용)

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

# 매핑 안 된 category가 있는지 확인 (있으면 즉시 알 수 있게)
unmapped = set(df["category"].unique()) - set(midcat_map.keys())
assert not unmapped, f"매핑 안 된 카테고리: {unmapped}"

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