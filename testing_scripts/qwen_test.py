# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import pandas as pd
# import torch
# from tqdm import tqdm
# import os
# import json

# # default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
# )

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
# images = os.listdir("/media/mlr_lab/325C37DE7879ABF2/AyushAnand/ml_challenge/teting")


# with open('/media/mlr_lab/325C37DE7879ABF2/AyushAnand/ml_challenge/download_status.json', 'r') as file:
#     image_mapping = json.load(file)  

# # print(image_mapping)   
# for i, (filename, entity_name) in tqdm(enumerate(image_mapping.items())):
#     if entity_name is not None:
#         # print(f"Image: {filename} -> Entity Name: {entity_name}")
#         messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": "/media/anil/ec5448df-0452-49b1-825b-d08ae2473211/Ashu/prarabdha/66e31d6ee96cd_student_resource_3/student_resource_3/teting/"+filename,
#                 },
#                 {"type": "text", "text": f"given this image just return me the {entity_name} of the item.Just return the numerical value along with dimension and please ensure that units/dimensions are always mentioned. If voltage/wattage is asked return the unit as volt/wattage ."}
#             ],
#         }
#     ]

#     # Preparation for inference
#         text = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#         inputs = inputs.to("cuda")

#         # Inference: Generation of the output
#         generated_ids = model.generate(**inputs, max_new_tokens=128)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         print(output_text)
#         with open("temp2.txt", "a") as f:
#             f.writelines(f"{i+1}: " + output_text[0]+"\n")

#     else:
#         print(f"Image: {filename} -> No entity name (broken image link)")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

# Load model and processor (move this outside the loop)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4")
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {
        'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'
    },
    'maximum_weight_recommendation': {
        'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'
    },
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {
        'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'
    }
}

# Load image mapping
with open('/media/mlr_lab/325C37DE7879ABF2/AyushAnand/ml_challenge/download_status.json', 'r') as file:
    image_mapping = json.load(file)
first_24 = int(0.24*len(image_mapping))
first_36 = int(0.36*len(image_mapping))
image_mapping = dict(islice(image_mapping.items(), first_24, first_36))
print(image_mapping)

def process_image(filename, entity_name):
    if entity_name is None:
        return filename, ""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"/media/mlr_lab/325C37DE7879ABF2/AyushAnand/ml_challenge/teting/{filename}",
                },
                {"type": "text", "text": f'''Given this image, return only the {entity_name} of the item in the format '{{number}} {{unit}}'. 
                 Use one of the units from the following list: {', '.join(entity_unit_map[entity_name])}.
                Ensure that the output strictly follows the format and uses the full unit names as specified.Don't use any abbreviations.'''}

                # {"type": "text", "text": f"Given this image, return only the {entity_name} of the item in the format '{{number}} {{unit}}'. Use one of the units from the following list: {', '.join(entity_unit_map[entity_name])}. If the entity is 'voltage' or 'wattage', return the value with the appropriate unit ('volt' or 'watt'). Ensure that the output strictly follows the format and units specified."}
                
                # {"type": "text", "text": f"given this image just return me the {entity_name} of the item. Just return the numerical value along with dimension and please ensure that units/dimensions are always mentioned. If voltage/wattage is asked return the unit as volt/wattage ."}
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
    return filename, output_text[0]

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=6) as executor:
    future_to_filename = {executor.submit(process_image, filename, entity_name): filename 
                          for filename, entity_name in image_mapping.items()}
    
    with open("24_to_36.txt", "w") as f:
        for future in tqdm(as_completed(future_to_filename), total=len(image_mapping)):
            filename = future_to_filename[future]
            try:
                result_filename, result_text = future.result()
                f.write(f"{result_filename}: {result_text}\n")
            except Exception as exc:
                print(f'{filename} generated an exception: {exc}')

print("Processing complete.")