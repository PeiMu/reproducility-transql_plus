-- Element-wise Arithmetic - Residual Add (paper section 3.2.1, Table 1 row 3)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `add_out`
SELECT a.row_index, a.chunk_index, list_transform(generate_series(1, len(a.v)), i -> CAST(a.v[i] + b.v[i] AS FLOAT)) AS v FROM residual a JOIN projection b ON a.row_index = b.row_index AND a.chunk_index = b.chunk_index;
