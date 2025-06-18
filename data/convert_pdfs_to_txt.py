import os
from pdfminer.high_level import extract_text

# === 📁 Set your input/output folders ===
input_folder = r"D:\projects\CamusGPT\data\training_datasets"
output_folder = r"D:\projects\CamusGPT\data\raw_texts"

# === 📂 Make sure output folder exists ===
os.makedirs(output_folder, exist_ok=True)

# === 🔁 Loop through all PDF files ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_folder, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)

        try:
            text = extract_text(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)
            print(f"✅ Converted: {filename} → {txt_filename}")
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")
