from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import pandas as pd
from tqdm import tqdm
import torch
df = pd.read_csv("test.csv")
model_id = "microsoft/Phi-3-vision-128k-instruct" 

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype=torch.float16, _attn_implementation='eager') # use _attn_implementation='eager' to disable flash attention

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 


for i in tqdm(range(len(df))):

    messages = [ 
        {"role": "user", "content": "<|image_1|>\n"+f"given this image just return me the {df.iloc[i]['entity_name']} of the item.Just return the numerical value along with dimension and please ensure that units/dimensions are always mentioned. If voltage/wattage is asked return the unit as volt/wattage ."},  
    ] 

    
    # url = df.iloc[i]["image_link"]
    # image = Image.open(requests.get(url, stream=True).raw) 

    url = df.iloc[i]["image_link"]
    image = Image.open(requests.get(url, stream=True).raw) 

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": 50, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

    print(response)

    with open("ans.txt","w") as f:
        f.writelines(response[0]+"\n")