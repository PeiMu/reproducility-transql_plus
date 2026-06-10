-- Normalization - RMSNorm (paper section 3.2.1, Table 1 row 5)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `norm_out_sq`
SELECT row_index, SUM(list_sum(list_transform(v, x -> x * x))) AS sq_sum FROM input GROUP BY row_index;

-- Step 2: creates table `norm_out`
SELECT n.row_index, n.chunk_index, list_transform(generate_series(1, len(n.v)), i -> CAST(n.v[i] / sqrt(s.sq_sum / 4.0 + 0.0000100000) * w.v[i] AS FLOAT)) AS v FROM input n JOIN norm_out_sq s ON n.row_index = s.row_index JOIN gamma w ON n.chunk_index = w.chunk_index;
