-- Post-opt section 4.2: Gate+Up Table Fusion
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `gate_gateup_w`
SELECT row_index, chunk_index, v, 'G' AS flag FROM gate_proj UNION ALL SELECT row_index, chunk_index, v, 'U' AS flag FROM up_proj;

-- Step 2: creates table `gate_gateup_dp`
SELECT a.row_index AS act_row, CASE w.flag WHEN 'G' THEN w.row_index WHEN 'U' THEN w.row_index + 8 END AS out_col, w.flag, SUM(list_dot_product(a.v, w.v)) AS val FROM norm2_out a JOIN gate_gateup_w w ON a.chunk_index = w.chunk_index GROUP BY a.row_index, w.row_index, w.flag;

-- Step 3: creates table `gate_gateup`
SELECT act_row AS row_index, CASE flag WHEN 'G' THEN out_col - (out_col % 2) WHEN 'U' THEN (out_col - 8) - ((out_col - 8) % 2) END AS chunk_index, array_agg(val ORDER BY out_col) AS v, flag FROM gate_gateup_dp GROUP BY act_row, flag, CASE flag WHEN 'G' THEN out_col - (out_col % 2) WHEN 'U' THEN (out_col - 8) - ((out_col - 8) % 2) END;
