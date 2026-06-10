-- RoPE - Rotary Positional Encoding (Decision D2, not in paper Table 1)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `q_rope`
SELECT q.row_index, q.chunk_index, list_transform(generate_series(1, 1), i -> CAST(q.v[2*i-1] * r.cos[i] - q.v[2*i] * r.sin[i] AS FLOAT)) AS v_even, list_transform(generate_series(1, 1), i -> CAST(q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i] AS FLOAT)) AS v_odd FROM q q JOIN rope r ON r.chunk_index = q.chunk_index AND r.row_index = q.row_index;
