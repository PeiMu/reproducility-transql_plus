-- QK Attention - MatMul variant with GQA + causal mask (Decision D4)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `qk_scores`
SELECT q.row_index AS q_tok, k.row_index AS k_tok, q.chunk_index // 2 AS head_id, SUM(list_dot_product(q.v_even, k.v_even) + list_dot_product(q.v_odd, k.v_odd)) AS score FROM q_rope q JOIN k_rope k ON q.chunk_index % 2 = k.chunk_index % 2 AND q.chunk_index // 4 = k.chunk_index // 2 AND k.row_index <= q.row_index GROUP BY q.row_index, k.row_index, q.chunk_index // 2;
