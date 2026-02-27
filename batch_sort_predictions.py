import os
import shutil
import time
import torch
from torchvision import transforms
from PIL import Image

from Training.model import PersonAttributeModel

# ---------- SETTINGS ----------
MODEL_PATH = "Training/multihead_model.pth"
INPUT_FOLDER = "/Users/ananth001gmail.com/Downloads/check_check"
OUTPUT_FOLDER = "/Users/ananth001gmail.com/Downloads/check_check_sorted_predictions"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# ---------- LOAD MODEL ----------
model = PersonAttributeModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------- LABEL MAPS ----------
hair_labels = {0: "no_hair", 1: "short_hair", 2: "long_hair"}
beard_labels = {0: "no_beard", 1: "short_beard", 2: "long_beard"}
eyewear_labels = {0: "no_glasses", 1: "glasses"}
gender_labels = {0: "female", 1: "male"}
hat_labels = {0: "no_hat", 1: "hat"}

# ---------- CREATE OUTPUT FOLDERS ----------
for attr, mapping in {
    "hair": hair_labels,
    "beard": beard_labels,
    "eyewear": eyewear_labels,
    "gender": gender_labels,
    "hat": hat_labels
}.items():
    
    for label in mapping.values():
        os.makedirs(os.path.join(OUTPUT_FOLDER, attr, label), exist_ok=True)

total_images = 0
total_inference_time = 0.0

for img_name in os.listdir(INPUT_FOLDER):

    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_FOLDER, img_name)
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():
        outputs = model(image_tensor)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    total_inference_time += (end - start)
    total_images += 1

    if total_images >= 100:
        break

    hair_pred = torch.argmax(outputs["hair"], dim=1).item()
    beard_pred = torch.argmax(outputs["beard"], dim=1).item()
    eyewear_pred = torch.argmax(outputs["eye"], dim=1).item()
    gender_pred = torch.argmax(outputs["gender"], dim=1).item()
    hat_pred = torch.argmax(outputs["hat"], dim=1).item()

    # ---------- PRINT OUTPUT ----------
    print(f"\nImage: {img_name}")
    print("READABLE OUTPUT")
    print("Beard:", beard_labels[beard_pred])
    print("Hair:", hair_labels[hair_pred])
    print("Eyewear:", eyewear_labels[eyewear_pred])
    print("Gender:", gender_labels[gender_pred])
    print("Hat:", hat_labels[hat_pred])

    print("RAW CLASS OUTPUT")
    print("Beard:", beard_pred)
    print("Hair:", hair_pred)
    print("Eyewear:", eyewear_pred)
    print("Gender:", gender_pred)
    print("Hat:", hat_pred)

    # ---------- COPY TO PREDICTED FOLDERS ----------
    shutil.copy(
        img_path,
        os.path.join(OUTPUT_FOLDER, "hair", hair_labels[hair_pred], img_name)
    )

    shutil.copy(
        img_path,
        os.path.join(OUTPUT_FOLDER, "beard", beard_labels[beard_pred], img_name)
    )

    shutil.copy(
        img_path,
        os.path.join(OUTPUT_FOLDER, "eyewear", eyewear_labels[eyewear_pred], img_name)
    )

    shutil.copy(
        img_path,
        os.path.join(OUTPUT_FOLDER, "gender", gender_labels[gender_pred], img_name)
    )

    shutil.copy(
        img_path,
        os.path.join(OUTPUT_FOLDER, "hat", hat_labels[hat_pred], img_name)
    )
avg_time = total_inference_time / total_images

print("\n===== PURE MODEL INFERENCE TIME =====")
print(f"Images processed: {total_images}")
print(f"Total forward-pass time: {total_inference_time:.4f} sec")
print(f"Average per image: {avg_time*1000:.4f} ms")

