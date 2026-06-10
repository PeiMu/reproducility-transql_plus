-- Normalization - Softmax, 4-step stable variant (not in paper)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `attn_weights_max`
SELECT q_tok, head_id, MAX(score) AS max_score FROM qk_scores GROUP BY q_tok, head_id;

-- Step 2: creates table `attn_weights_exp`
SELECT s.q_tok, s.k_tok, s.head_id, EXP(s.score - m.max_score) AS exp_val FROM qk_scores s JOIN attn_weights_max m ON s.q_tok = m.q_tok AND s.head_id = m.head_id;

-- Step 3: creates table `attn_weights_sum`
SELECT q_tok, head_id, SUM(exp_val) AS sum_exp FROM attn_weights_exp GROUP BY q_tok, head_id;

-- Step 4: creates table `attn_weights`
SELECT e.q_tok, e.k_tok, e.head_id, CAST(e.exp_val / s.sum_exp AS FLOAT) AS attn_weight FROM attn_weights_exp e JOIN attn_weights_sum s ON e.q_tok = s.q_tok AND e.head_id = s.head_id;
