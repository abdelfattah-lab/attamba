## WIKITEXT2
(Pdb) tokens.shape
torch.Size([1, 1023])
(Pdb) self.cu_seqlens
tensor([   0, 1023], device='cuda:0', dtype=torch.int32)


(Pdb) self.ssm_prefill_tok_id
tensor([[0, 0, 0,  ..., 0, 0, 0]], device='cuda:0', dtype=torch.int32)
(Pdb) self.ssm_prefill_tok_id.shape
torch.Size([1, 1023])
(Pdb) self.ssm_prefill_tok_id.sum() 
tensor(0, device='cuda:0')
(Pdb) self.prefill_mask
BlockMask(
    kv_num_blocks=torch.Size([1, 1, 8]),
    kv_indices=torch.Size([1, 1, 8, 8]),
    full_kv_num_blocks=torch.Size([1, 1, 8]),
    full_kv_indices=torch.Size([1, 1, 8, 8]),
    q_num_blocks=torch.Size([1, 1, 8]),
    q_indices=torch.Size([1, 1, 8, 8]),
    full_q_num_blocks=torch.Size([1, 1, 8]),
    full_q_indices=torch.Size([1, 1, 8, 8]),
    BLOCK_SIZE=(128, 128),
    shape=(1, 1, 1024, 1024),
    sparsity=43.75%,
    mask_mod=doc_mask_mod
)
(Pdb) self.prefill_mask.to_dense()
tensor([[[[1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1]]]], device='cuda:0', dtype=torch.int32)
(Pdb) self.prefill_tok_id
tensor([   0,    1,    2,  ..., 1020, 1021, 1022], device='cuda:0')

## PIQA
(Pdb) tokens.shape
torch.Size([1, 973])
(Pdb) self.cu_seqlens
tensor([  0,  70, 140, 157, 177, 214, 248, 272, 297, 317, 338, 363, 389, 472,
        555, 601, 644, 697, 751, 777, 803, 827, 853, 879, 939, 973],
       device='cuda:0', dtype=torch.int32)
(Pdb) self.ssm_prefill_tok_id
tensor([[ 0,  0,  0,  .... 1, 1, 1, 1...., 2, 2, 2, 2, ....., 23, 23, 23, ...., 24]], device='cuda:0', dtype=torch.int32)
(Pdb) self.ssm_prefill_tok_id.shape
torch.Size([1, 973])
(Pdb) self.ssm_prefill_tok_id.sum() 
tensor(11572, device='cuda:0')
(Pdb) self.prefill_mask
BlockMask(
    kv_num_blocks=torch.Size([1, 1, 8]),
    kv_indices=torch.Size([1, 1, 8, 8]),
    full_kv_num_blocks=torch.Size([1, 1, 8]),
    full_kv_indices=torch.Size([1, 1, 8, 8]),
    q_num_blocks=torch.Size([1, 1, 8]),
    q_indices=torch.Size([1, 1, 8, 8]),
    full_q_num_blocks=torch.Size([1, 1, 8]),
    full_q_indices=torch.Size([1, 1, 8, 8]),
    BLOCK_SIZE=(128, 128),
    shape=(1, 1, 1024, 1024),
    sparsity=68.75%,
    mask_mod=doc_mask_mod
)
(Pdb) self.prefill_mask.to_dense()
tensor([[[[1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 1]]]], device='cuda:0', dtype=torch.int32)
(Pdb) self.prefill_tok_id
tensor([ 0,  1,  2, ..., 21, 22, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,  0, 1, 2, 3....], device='cuda:0')
