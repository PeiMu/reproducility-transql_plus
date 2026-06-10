-- Matrix Multiplication (paper section 3.2.1, Table 1 row 1)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `matmul_out_dp`
SELECT a.row_index AS act_row, w.row_index AS out_col, SUM(list_dot_product(a.v, w.v)) AS val FROM activation a JOIN weight w ON a.chunk_index = w.chunk_index GROUP BY a.row_index, w.row_index;

-- Step 2: creates table `matmul_out`
SELECT act_row AS row_index, out_col - (out_col % 2) AS chunk_index, array_agg(val ORDER BY out_col) AS v FROM matmul_out_dp GROUP BY act_row, out_col - (out_col % 2);
