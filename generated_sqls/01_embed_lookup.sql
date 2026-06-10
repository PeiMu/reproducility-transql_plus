-- Embedding Lookup (paper section 3.2.2, lookup table)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `x_0`
SELECT t.pos AS row_index, e.chunk_index, e.v FROM input_tokens t JOIN embed_tokens e ON t.token_id = e.row_index ORDER BY t.pos, e.chunk_index;
