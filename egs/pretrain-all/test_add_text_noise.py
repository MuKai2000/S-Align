import torch

"""text_tokens = torch.randint(1, 20, (5, 10))
print(text_tokens.shape, text_tokens)

add_len = int(text_tokens.shape[-1] * 0.5)
print(add_len)

new_text_tokens = torch.zeros(text_tokens.shape[0], text_tokens.shape[-1] + add_len)
print(new_text_tokens.shape, new_text_tokens)

add_mask, _ = torch.sort(torch.randint(0, text_tokens.shape[-1], (text_tokens.shape[0], add_len)), dim=1)
print(add_mask.shape, add_mask)

print(text_tokens[:,add_mask[-1]:])

for i in reversed(range(1, add_len)):
    print(i)"""

TIME = 20000.0
END = 45000.0
for update_num in range(0, 50000):
    rate = 0.5 + 0.5 * min(1.0, (END - min(END, update_num)) / (END - TIME))
    print(update_num, rate)