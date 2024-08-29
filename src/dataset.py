import torch

class GPT_XDataset(torch.utils.data.Dataset):
    def __init__(self, text_data, seq_length=128):
        super(GPT_XDataset, self).__init__()

        self.seq_length = seq_length

        unique_chars = sorted(list(set(text_data)))
        self.text_size = len(text_data)
        self.num_chars = len(unique_chars)

        self.char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
        self.index_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        self.text_data = text_data

    def vocab_size(self):
        return self.num_chars
    
    def sequence_length(self):
        return self.seq_length
    
    def __len__(self):
        return self.text_size - self.seq_length
    
    def __getitem__(self, index):
        text_chunk = self.text_data[index: index + self.seq_length + 1]
        indices = [self.char_to_index[char] for char in text_chunk]
        input_tensor = torch.tensor(indices[:-1], dtype=torch.long)
        target_tensor = torch.tensor(indices[1:], dtype=torch.long)
        return input_tensor, target_tensor
