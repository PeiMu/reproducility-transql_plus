-- Attention x V - MatMul variant
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `attn_out_vs`
SELECT row_index AS tok, chunk_index, unnest(generate_series(0, 1)) AS elem_pos, CAST(unnest(v) AS FLOAT) AS val FROM v;

-- Step 2: creates table `attn_out_w`
SELECT s.q_tok, s.head_id * 2 + v.chunk_index % 2 AS out_chunk_index, v.elem_pos, CAST(SUM(s.attn_weight * v.val) AS FLOAT) AS val FROM attn_weights s JOIN attn_out_vs v ON s.k_tok = v.tok AND s.head_id // 2 = v.chunk_index // 2 GROUP BY s.q_tok, s.head_id, v.chunk_index, v.elem_pos;

-- Step 3: creates table `attn_out`
SELECT q_tok AS row_index, out_chunk_index AS chunk_index, array_agg(val ORDER BY elem_pos) AS v FROM attn_out_w GROUP BY q_tok, out_chunk_index;
