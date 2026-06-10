-- Post-opt section 4.2: QKV Table Fusion (UNION ALL + flag column)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `q_qkv_w`
SELECT row_index, chunk_index, v, 'Q' AS flag FROM q_proj UNION ALL SELECT row_index, chunk_index, v, 'K' AS flag FROM k_proj UNION ALL SELECT row_index, chunk_index, v, 'V' AS flag FROM v_proj;

-- Step 2: creates table `q_qkv_dp`
SELECT a.row_index AS act_row, CASE w.flag WHEN 'Q' THEN w.row_index WHEN 'K' THEN w.row_index + 4 WHEN 'V' THEN w.row_index + 6 END AS out_col, w.flag, SUM(list_dot_product(a.v, w.v)) AS val FROM norm1_out a JOIN q_qkv_w w ON a.chunk_index = w.chunk_index GROUP BY a.row_index, w.row_index, w.flag;

-- Step 3: creates table `q_qkv`
SELECT act_row AS row_index, CASE flag WHEN 'Q' THEN out_col - (out_col % 2) WHEN 'K' THEN (out_col - 4) - ((out_col - 4) % 2) WHEN 'V' THEN (out_col - 6) - ((out_col - 6) % 2) END AS chunk_index, array_agg(val ORDER BY out_col) AS v, flag FROM q_qkv_dp GROUP BY act_row, flag, CASE flag WHEN 'Q' THEN out_col - (out_col % 2) WHEN 'K' THEN (out_col - 4) - ((out_col - 4) % 2) WHEN 'V' THEN (out_col - 6) - ((out_col - 6) % 2) END;
