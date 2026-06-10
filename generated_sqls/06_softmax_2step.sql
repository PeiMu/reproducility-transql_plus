-- Normalization - Softmax, 2-step paper-faithful (Decision D5)
-- Demo dims: chunk_size=2, hidden_dim=4, kv_dim=2, ffn_dim=8

-- Step 1: creates table `attn_weights_expsum`
SELECT q_tok, head_id, SUM(exp(score)) AS summation FROM qk_scores GROUP BY q_tok, head_id;

-- Step 2: creates table `attn_weights`
SELECT s.q_tok, s.k_tok, s.head_id, CAST(exp(s.score) / e.summation AS FLOAT) AS attn_weight FROM qk_scores s JOIN attn_weights_expsum e ON s.q_tok = e.q_tok AND s.head_id = e.head_id;
