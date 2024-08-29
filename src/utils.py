def get_pairs(word):
    setter = set()
    first = word[0]
    for i in word[1:]:
        setter.add((first, i))
        first = i

    return setter



def byte_pair_encoding(self, token):
    if token in self.cache:
        return self.cache[token]
    



    
    