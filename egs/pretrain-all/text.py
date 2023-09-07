import torch
mt_encoder_output = torch.rand(5,3,7)  # [len, bs, dim]
mt_encoder_padding_mask = torch.Tensor(
    [
        [False, False, False, False, True],
        [False, False, False, True, True],
        [False, False, False, False, False],
    ]
).bool()
print(mt_encoder_output.shape, mt_encoder_padding_mask.shape)
print(mt_encoder_output, mt_encoder_padding_mask)

encoder_output = mt_encoder_output.transpose(0,1).reshape(-1, mt_encoder_output.shape[-1])

encoder_padding_mask = mt_encoder_padding_mask.reshape(-1)

print(encoder_output.shape, encoder_padding_mask.shape)
print(encoder_output, encoder_padding_mask)

encoder_output = encoder_output[~encoder_padding_mask]
print(encoder_output.shape)
print(encoder_output)
