import os
import json
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# Set up device and data type
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Florence model and processor
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True
)

# Load the image mapping from the JSON file
with open('/media/mlr_lab/325C37DE7879ABF2/AyushAnand/ml_challenge/download_status.json', 'r') as file:
    image_mapping = json.load(file)

# Loop through each image and its corresponding entity name
for i, (filename, entity_name) in tqdm(enumerate(image_mapping.items())):
    if entity_name is not None:
        # Load and preprocess the image
        image_path = os.path.join(
            "/media/mlr_lab/325C37DE7879ABF2/AyushAnand/ml_challenge/teting",
            filename
        )
        image = Image.open(image_path).convert("RGB")

        # Prepare the prompt for the Florence model
        prompt = (
            f"<ImageQuestion>"
            f"Given this image, just return me the {entity_name} of the item. "
            f"Just return the numerical value along with dimension and please ensure that units/dimensions are always mentioned. "
            f"If voltage/wattage is asked, return the unit as volt/wattage."
            f"<ImageQuestion>"
        )

        # Prepare inputs for the model
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(device, torch_dtype)

        # Generate the output
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            do_sample=False,
            num_beams=3,
        )

        # Decode the generated text
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)

        # Print and save the output
        print(generated_text)
        with open("temp2.txt", "a") as f:
            f.writelines(f"{i+1}: {generated_text}\n")
    else:
        print(f"Image: {filename} -> No entity name (broken image link)")
