-- Post-opt section 4.3: ROW2COL Pivoted MatMul (PIVOT -> CROSS JOIN -> POSITIONAL JOIN)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `pivot_out_act_piv`
PIVOT activation ON chunk_index IN (0, 2) USING first(v) GROUP BY row_index;

-- Step 2: creates table `weight_piv`
PIVOT weight ON chunk_index IN (0, 2) USING first(v) GROUP BY row_index;

-- Step 3: creates table `pivot_out_dp_sq0`
SELECT a.row_index AS act_row, w.row_index AS out_col, list_dot_product(a."0", w."0") AS v0 FROM pivot_out_act_piv a CROSS JOIN weight_piv w ORDER BY a.row_index, w.row_index;

-- Step 4: creates table `pivot_out_dp_sq1`
SELECT a.row_index AS act_row, w.row_index AS out_col, list_dot_product(a."2", w."2") AS v1 FROM pivot_out_act_piv a CROSS JOIN weight_piv w ORDER BY a.row_index, w.row_index;

-- Step 5: creates table `pivot_out_dp`
SELECT pivot_out_dp_sq0.act_row, pivot_out_dp_sq0.out_col, pivot_out_dp_sq0.v0 + pivot_out_dp_sq1.v1 AS val FROM pivot_out_dp_sq0 POSITIONAL JOIN pivot_out_dp_sq1;

-- Step 6: creates table `pivot_out`
SELECT act_row AS row_index, out_col - (out_col % 2) AS chunk_index, array_agg(val ORDER BY out_col) AS v FROM pivot_out_dp GROUP BY act_row, out_col - (out_col % 2);
