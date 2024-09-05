from pathlib import Path

def get_config(lora=False):

    if lora is False:
        config = {
                # Data
                "dataset": "Salesforce/wikitext",
                "subset": "wikitext-2-raw-v1",
                "seq_len": 350,
                "train_size" : 1,

                # Model
                "model": "openai-community/gpt2",
                "batch_size": 14,
                "gradient_accumulation": True,
                "accumulation_iter": 5,
                "num_epochs": 8,
                "lr": 1e-4,

                # Saving model
                "model_folder": "weights",
                "model_basename": "gpt_no_lora_",
                "preload" : None,
                "experiment_name": "runs/gpt_no_lora",

                # LoRA
                "lora": False
            }
    else:
        config = {
                # Data
                "dataset": "Salesforce/wikitext",
                "subset": "wikitext-2-raw-v1",
                "seq_len": 350,
                "train_size" : 1,

                # Model
                "model": "openai-community/gpt2",
                "batch_size": 14,
                "gradient_accumulation": True,
                "accumulation_iter": 5,
                "num_epochs": 8,
                "lr": 1e-3,

                # Saving model
                "model_folder": "weights",
                "model_basename": "gpt_lora_all_layers_",
                "preload" : None,
                #### gpt_lora_all_layers / gpt_lora_no_output / gpt_lora_no_ff / gpt_lora_no_proj
                "experiment_name": "runs/gpt_lora_all_layers", 

                # LoRA
                "lora": True,
                #### 'attention' / 'feed_forward' / 'output' / 'projection'
                "target_modules": ['attention' / 'feed_forward' / 'output' / 'projection'],
                "rank": 16,
                "lora_alpha": 32,
                "lora_dropout" : 0.1
            }

    return config


def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
