from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_tokenizer(config):
    return AutoTokenizer.from_pretrained(config['model'])

def get_model(config):
    model = AutoModelForCausalLM.from_pretrained(config['model'])

    if config['lora']:
        target_modules = []
        if 'attention' in config['target_modules']:
            target_modules.append('c_attn')

        if 'feed_forward' in config['target_modules']:
            target_modules.append('c_fc')

        if 'output' in config['target_modules']:
            target_modules.append('lm_head')

        if 'projection' in config['target_modules']:
            target_modules.append('c_proj')

        peft_config = LoraConfig(task_type="CAUSAL",
                                 inference_mode=False,
                                 r=config['rank'],
                                 lora_alpha=config['lora_alpha'],
                                 lora_dropout=config['lora_dropout'],
                                 target_modules= target_modules,
                                 fan_in_fan_out=True
                                )
        model = get_peft_model(model, peft_config)
        
    return model 
