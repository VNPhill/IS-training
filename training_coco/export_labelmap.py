import os

from config import ACTIVE_CLASSES, MODEL_TYPE

output_folder = "./exports"
os.makedirs(output_folder, exist_ok=True)

with open(f"{output_folder}/labelmap_{MODEL_TYPE}.txt", "w") as f:
    f.write("background\n")
    for cls in ACTIVE_CLASSES:
        f.write(cls + "\n")
        
        