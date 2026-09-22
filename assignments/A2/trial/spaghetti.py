import os, re
from pathlib import Path
from tokenizers import pre_tokenizers, Tokenizer, models, trainers


base_dir = Path(__file__).parent.parent                    # trial/ -> A2/
with open(base_dir / "data" / "hobbit.txt", encoding="utf-8") as f:
    hobbit = f.read()
print("the hobbit text is:", len(hobbit))
with open(base_dir / "data" / "lostworld.txt", encoding="utf-8") as f:
    lostworld = f.read()
print("the lost world text is:", len(lostworld))


sample = hobbit[5700:5800]
# sample = 'He said "no!" twice.'

# print(repr(sample), "\n")

splitter = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation(),
    # pre_tokenizers.Whitespace(),
])

# results = splitter.pre_tokenize_str(sample)
# # print("cut at spaces and punctuation:", results, "\n")

# pieces = []
# for item in results:
#     pieces.append(item[0])

# print("cut at spaces only:", pieces, "\n")

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = splitter

trainer = trainers.BpeTrainer(
    vocab_size=500,
    special_tokens=["<unk>", "<s>", "</s>"],
    show_progress=False,
)

tokenizer.train_from_iterator([hobbit, lostworld], trainer)

# print("size:", tokenizer.get_vocab_size())
# print("<unk> is", tokenizer.token_to_id("<unk>"))
# print("<s> is", tokenizer.token_to_id("<s>"))
# print("</s> is", tokenizer.token_to_id("</s>"))

encoded = tokenizer.encode("Bilbo went home.")

# print(encoded.tokens)
# print(encoded.ids)

numbers = tokenizer.encode("Bilbo went home.").ids

BOS = 1
EOS = 2

for_bigram = [BOS] + numbers + [EOS]
for_trigram = [BOS, BOS] + numbers + [EOS]

print("bigram: ", for_bigram)
print("trigram:", for_trigram)