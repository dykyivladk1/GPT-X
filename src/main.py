from dataset import GPT_XDataset
from model import GPT_X

import torch
import polip 
import argparse

import os



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = '/Users/vladyslav/Downloads/sample.txt')
parser.add_argument('--max_iters', type = int, default = 10000)
parser.add_argument('--block_size', type = int, default = 128)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--learning_rate', type = float, default = 3e-4)
parser.add_argument('--betas', type = tuple, default = (0.9, 0.95))
parser.add_argument('--weight_decay', type = float, default = 0.1)
parser.add_argument('--grad_norm_clip', type = float, default = 1.0)
parser.add_argument('--mode', type = str, default = 'training')
parser.add_argument('--context', type = str, default = None)
parser.add_argument('--weights_path', type = str, default = None)
parser.add_argument('--max_tokens', type = int, default = 500)
args = parser.parse_args()



with open(args.data_path, 'r') as file:
    content = file.read()


block_size = args.block_size
max_iters = args.max_iters
batch_size = args.batch_size
learning_rate = args.learning_rate
betas = args.betas
weight_decay = args.weight_decay
grad_norm_clip = args.grad_norm_clip





dataset = GPT_XDataset(content, block_size)

vocab_size = dataset.vocab_size()



device = polip.decider()


model = GPT_X(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embed=128).to(device)



optimizer = model.set_optimizers_(learning_rate, betas, weight_decay)

train_dataset = dataset

train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler = torch.utils.data.RandomSampler(train_dataset, replacement = True, num_samples = int(1e10)),
    shuffle = False, pin_memory = True, batch_size = batch_size, num_workers = 0
)


model.train()
if args.mode == 'training':
    for iteration, batch in enumerate(train_loader, start=1):
        print(f"Iteration: {iteration}")

        batch = [t.to(device) for t in batch]
        x, y = batch
        logits, loss = model(x, y)

        print(f"Loss computed: {loss.item()}")

        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()

        if iteration >= max_iters:
            break

    torch.save(model.state_dict(), 'weights/model.pth')


elif args.mode == 'test':

    context = args.context
    device = polip.decider('cpu')
    model = model.to(device)


    if os.path.exists(args.weights_path):
        model.load_state_dict(torch.load(args.weights_path))
        print('Model Weights Loaded!')
    else:
        model = model


    x = torch.tensor([train_dataset.char_to_index[s] for s in context], dtype=torch.long)[None,...].to(device)
    y = model.generate(x, args.max_tokens, temperature=1.0, do_sample=True, top_k=10)[0]
    completion = ''.join([train_dataset.index_to_char[int(i)] for i in y])
    print(completion)