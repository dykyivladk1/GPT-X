from utils import bytes_to_unicode, get_pairs

import regex as re



class Encoder:
    def __init__(self, encoder, byte_pair_encoding_merges):
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_pair_encoding_ranks = dict(zip(byte_pair_encoding_merges, range(len(byte_pair_encoding_merges))))
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bytes_to_unicode(self):
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def byte_pair_encoding(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.byte_pair_encoding_ranks.get(pair, float('inf')))
            if bigram not in self.byte_pair_encoding_ranks:
                break
            first, second = bigram

            new_word = []
            i = 0
            while i < len(word):
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        byte_pair_encoding_idx = []
        tokens = re.findall(self.pat, text)

        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.byte_pair_encoding(token_translated).split(' ')
            token_idx = [self.encoder[byte_pair_encoding_token] for byte_pair_encoding_token in token_merged]
            byte_pair_encoding_idx.extend(token_idx)

        return byte_pair_encoding_idx

    def decode(self, byte_pair_encoding_idx):
        tokens_merged = [self.decoder[token] for token in byte_pair_encoding_idx]
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        text = tokens_bytes.decode('utf-8', errors='replace')
        return text
