import os
import torch
from typing import Optional
from safetensors import safe_open
from safetensors.torch import load_file, save_file 

def load_and_save(load:dict, output_dir:str, start:Optional[int], end:Optional[int], attr_type:Optional[str] = None):
    if attr_type is None:    
        for i in range(start,end):
            layer_weights = {k.replace(f"model.layers.{i}","layers.0"): v for k, v in load.items() if f"layers.{i}." in k}
            
            if len(layer_weights):  # Ensure there are weights to save
                out_file = f"{output_dir}/separated_weights/layers{i}.safetensors"
                save_file(layer_weights, out_file)
                print(f"Saved layer {i} weights to {out_file}")
            else:
                print(f"No weights found for layer {i}")
    else:    
        layer_weights = {k.replace("model.",""): v for k, v in load.items() if f"{attr_type}" in k and "31" not in k}
        if len(layer_weights):  # Ensure there are weights to save
            out_file = f"{output_dir}/separated_weights/{attr_type}.safetensors"
            save_file(layer_weights, out_file)
            print(f"Saved {attr_type} weights to {out_file}")
        else:
            print(f"No weights found for layer {attr_type}")


def merge_safe_tensors(num_layers: int,model_dir: str, output_dir: Optional[str] = None):
    if output_dir is None:
        output_dir = model_dir
    load1 = load_file(model_dir + "/model-00001-of-00004.safetensors")
    os.makedirs(f"{output_dir}/separated_weights", exist_ok=True)
    load_and_save(load1,output_dir,None,None,"embed_tokens") 
    load_and_save(load1,output_dir,0,9)    
    del load1
    load2 = load_file(model_dir + "/model-00002-of-00004.safetensors")
    load_and_save(load2,output_dir,9,20) 
    load3 = load_file(model_dir + "/model-00003-of-00004.safetensors")
    load23 = {**load2, **load3}
    load_and_save(load23,output_dir,20,21)    
    del load2
    del load23
    load_and_save(load3,output_dir,21,31)    
    load4 = load_file(model_dir + "/model-00004-of-00004.safetensors")
    load34 = {**load3, **load4}
    load_and_save(load34,output_dir,31,num_layers)    
    del load3
    del load34
    load_and_save(load4,output_dir,None,None,"lm_head")
    load_and_save(load4,output_dir,None,None,"norm")  
    del load4
    
if __name__ == '__main__':
    #merge_safe_tensors(32,"C:/yoghes/llms/Llama-3.1-8B-Instruct")
    load = load_file("C:/yoghes/llms/Llama-3.1-8B-Instruct" + f"/separated_weights/embed_tokens.safetensors") 
    for i in list(load.keys()):
                print(i)
    for i in range(0,32):
        load = load_file("C:/yoghes/llms/Llama-3.1-8B-Instruct" + f"/separated_weights/layers{i}.safetensors") 
        if len(list(load.keys())) != 9:
            print('wtf')
            break
        for i in list(load.keys()):
                print(i)
    load = load_file("C:/yoghes/llms/Llama-3.1-8B-Instruct" + f"/separated_weights/lm_head.safetensors") 
    for i in list(load.keys()):
                print(i)
    load = load_file("C:/yoghes/llms/Llama-3.1-8B-Instruct" + f"/separated_weights/norm.safetensors") 
    for i in list(load.keys()):
                print(i)    
   
    