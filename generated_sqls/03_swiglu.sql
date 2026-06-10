-- Element-wise Function - SwiGLU (paper section 3.2.1, Table 1 row 2)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `swiglu_out`
SELECT g.row_index, g.chunk_index, list_transform(generate_series(1, len(g.v)), i -> CAST((g.v[i] / (1.0 + exp(-g.v[i]))) * u.v[i] AS FLOAT)) AS v FROM gate g JOIN up u ON g.row_index = u.row_index AND g.chunk_index = u.chunk_index;
