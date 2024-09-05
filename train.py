import time
import datetime
import warnings
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from model import get_model, get_tokenizer
from dataset import get_ds
from config import get_config, get_weights_file_path


parser = argparse.ArgumentParser()
parser.add_argument('-l','--lora', action='store_true', help='Train only LoRA model')
args = parser.parse_args()


def train(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    model = get_model(config).to(device)
    tokenizer = get_tokenizer(config)

    train_dataloader = get_ds(config, tokenizer, 'train')
    val_dataloader = get_ds(config, tokenizer, 'validation')

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], eps = 1e-9)


    num_training_steps = len(train_dataloader) * config['num_epochs']

    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters:,}")
    writer.add_scalar('n_parameters', n_parameters)

    # ========================================
    #               Training
    # ========================================
    t0 = time.time()
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Training epoch {epoch:02d}')
        epoch_loss = 0

        for batch in batch_iterator:
            input_tokens = batch['input_tokens'].to(device)
            mask = batch['mask'].to(device)

            output = model(input_tokens, attention_mask=mask, labels=input_tokens)
            loss = output.loss
            epoch_loss+= loss.item()

            batch_iterator.set_postfix({'Loss': f'{loss.item():6.3f}'})

            writer.add_scalar('train loss', loss.item(), global_step)

            if config['gradient_accumulation']:
                loss = loss / config['accumulation_iter']
                loss.backward()
                if global_step % config['accumulation_iter'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1

        total_epoch_loss = epoch_loss / len(batch_iterator)
        batch_iterator.write(f'EPOCH LOSS: {total_epoch_loss}')
        writer.add_scalar('epoch train loss', total_epoch_loss, global_step)

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)


        # ========================================
        #               Validation
        # ========================================
        batch_iterator = tqdm(val_dataloader, desc=f'Validation epoch {epoch:02d}')
        epoch_loss = 0
        model.eval()

        for batch in batch_iterator:
            input_tokens = batch['input_tokens'].to(device)
            mask = batch['mask'].to(device)

            with torch.no_grad():
                output = model(input_tokens, attention_mask=mask, labels=input_tokens)
                loss = output.loss
            epoch_loss+= loss.item()

            batch_iterator.set_postfix({'Val_Loss': f'{loss.item():6.3f}'})
            writer.add_scalar('val loss', loss.item(), global_step)

        batch_iterator.write(f'EPOCH VAL LOSS: {epoch_loss / len(batch_iterator)}')

        total_epoch_loss = epoch_loss / len(batch_iterator)
        writer.add_scalar('epoch val loss', total_epoch_loss, global_step)

        batch_iterator.write('=================================================')

    curr = time.time()
    elapsed = curr - t0

    total_time = str(datetime.timedelta(seconds=int(round((elapsed)))))
    writer.add_scalar('Time elapsed', round((elapsed)))
    print(f"Training took {total_time}")

    allocated = torch.cuda.max_memory_allocated(device='cuda:0') // 1024**2
    writer.add_scalar('Allocated memory', allocated)
    print(f"Total memory allocated {allocated}")

    # ========================================
    #               Testing
    # ========================================
    test_dataloader = get_ds(config, tokenizer, 'test')
    batch_iterator = tqdm(test_dataloader, desc='Test epoch')
    epoch_loss = 0
    model.eval()

    for batch in batch_iterator:
        input_tokens = batch['input_tokens'].to(device)
        mask = batch['mask'].to(device)

        with torch.no_grad():
            output = model(input_tokens, attention_mask=mask, labels=input_tokens)
            loss = output.loss
        epoch_loss+= loss.item()
        batch_iterator.set_postfix({'Test_loss': f'{loss.item():6.3f}'})

    mean_loss = epoch_loss / len(batch_iterator)
    perplexity = torch.exp(torch.tensor(mean_loss))

    batch_iterator.write(f'TEST LOSS: {mean_loss}')
    batch_iterator.write(f'PERPLEXITY: {perplexity}')


    writer.add_scalar('test loss', mean_loss)
    writer.add_scalar('test perplexity', perplexity)

if __name__ == '__main__':
    torch.manual_seed(42)
    warnings.filterwarnings('ignore')
    print("Training LoRA: ", args.lora)
    config = get_config(lora=args.lora)
    train(config)
