import pandas as pd

DATA_DIR = "/user/gaad2403/bachelor_thesis/anuraset/anuraset"
metadata = pd.read_csv(f"{DATA_DIR}/metadata.csv")

SPECIES_COLS = [
    "SPHSUR","BOABIS","SCIPER","DENNAH","LEPLAT","RHIICT","BOALEP","BOAFAB",
    "PHYCUV","DENMIN","ELABIC","BOAPRA","DENCRU","BOALUN","BOAALB","PHYMAR",
    "PITAZU","PHYSAU","LEPFUS","DENNAN","PHYALB","LEPLAB","SCIFUS","BOARAN",
    "SCIFUV","AMEPIC","LEPPOD","ADEDIP","ELAMAT","PHYNAT","LEPELE","RHISCI",
    "SCINAS","LEPNOT","ADEMAR","BOAALM","PHYDIS","RHIORN","LEPFLA","SCIRIZ",
    "DENELE","SCIALT"
]

species_number = metadata[SPECIES_COLS].sum(axis=1)
labeled = metadata[species_number > 0]   # unknown(배경음) 제외

print(f"라벨 있는 전체 샘플: {len(labeled)}")
print(f"그중 multi-label(2종 이상 동시발생) 샘플: {(species_number[species_number>0] > 1).sum()}")
print(f"비율: {(species_number[species_number>0] > 1).mean()*100:.1f}%")
print()
print("종 개수별 분포:")
print(species_number[species_number>0].value_counts().sort_index())