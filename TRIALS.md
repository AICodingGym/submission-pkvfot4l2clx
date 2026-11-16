# Trial record

Scores are inspiratory mean absolute error on the locked holdout: whole breaths, seed 42, 20% validation (13,581 breaths, 412,309 scored rows). Lower is better. A row counts only when `u_out` is 0. No trial has been submitted to the leaderboard.

Runtime is model training plus validation prediction. It does not include loading the CSV or building features.

| Trial | Model | Validation MAE | Runtime (s) |
| --- | --- | ---: | ---: |
| Reference, inside trial 1 | Constant pressure 17.595 | 7.594 | — |
| 1 | LightGBM, raw columns | 3.998 | 21.1 |
| 2 | LightGBM, lags 1–3 and cumulative sum | 1.375 | 27.2 |
| 3 | LightGBM, lags 1–20 and one-step change | 1.268 | 37.2 |
| 4 | Forward LSTM, 10 epochs | 1.583 | 21.3 |
| 5 | Forward LSTM, continued to epoch 200 | 0.901 | 390.7 |
| 6 | Bidirectional LSTM, 1 layer, 200 epochs | 0.768 | 767.8 |
| 7 | Bidirectional LSTM, 2 layers, hidden 128, 200 epochs | 0.552 | 1622.9 |
| 8 | Bidirectional LSTM, 4 layers, hidden 256, 200 epochs | 0.369 | 6813.8 |
| 9 | Trial 8 network, engineered features, cosine decay | 0.244 | 6735.8 |
| 10 | Trial 9 resumed, R=50 loss doubled, cosine peak 0.0003 | 0.222 | 6722.2 |
| 11 | Trial 10 fine-tuned on R=50 breaths, stitched back in | 0.220 | 5297.2 |
| B | New seed of trial 9, then the trial 10 R=50 upweight | 0.250 | 13286.4 |
| C | New seed, 4× loss on R=50 with C=20 or C=50 | 0.283 | 6599.2 |

The best single model is still trial 11, at 0.220. The best score so far is the average of trial 11 and model B, at 0.218. Exact numbers are in `experiments/`. The backward pass from trial 6 on, and the `u_in * R` and cumulative `u_in / C` features from trial 9 on, are outside ideas; see the flag at the end of this file.

## Trial 1 — raw columns

LightGBM, L1 loss, 200 trees, learning rate 0.1, 63 leaves. Each row was a separate example. Features were `R`, `C`, `time_step`, `u_in`, and `u_out`.

A constant prediction of the training-set mean inspiratory pressure, 17.595, scored 7.594 on the same rows. The tree scored 3.998.

The pipeline, the breath split, and the metric work. The current row alone is a weak description of pressure. The next trial added recent history from the same breath.

Log: `experiments/baseline.json`. Script: `baseline.py`.

## Trial 2 — short within-breath history

Same LightGBM settings as trial 1. Inside each breath, the new columns were `u_in` lagged 1, 2, and 3 steps, the cumulative sum of `u_in`, and a step index from 0 to 79. `R` and `C` were treated as categories. Three lags cover about a tenth of a second. That length was a short first test of history, not a tuned window. The cumulative sum was there to carry anything older.

Validation MAE fell from 3.998 to 1.375.

Every scored row is in the first 40 steps. After that, `u_out` is 1, so those rows are outside the MAE. The error is uneven:

| Group | MAE |
| --- | ---: |
| R = 5 | 0.892 |
| R = 20 | 1.349 |
| R = 50 | 1.780 |
| C = 10 | 1.755 |
| C = 20 | 1.260 |
| C = 50 | 1.048 |

Recent `u_in` history matters a lot. High resistance and low compliance are the hard breaths. The scored part of a breath is about 40 steps, so the next trial asked whether a longer lag list would keep helping.

Log: `experiments/features.json`. Script: `experiment_features.py`.

## Trial 3 — longer `u_in` history

Same LightGBM settings and the same features as trial 2, except `u_in` lags were extended from 1–3 to 1–20 and the one-step change in `u_in` was added.

Validation MAE moved from 1.375 to 1.268, a gain of 0.107.

The tree mostly used the cumulative sum, `time_step`, `R`, `C`, the step index, and the most recent lags. Lags past about 5 steps were weak. `u_out` was almost unused, which fits the metric: on scored rows it is always 0. High resistance was still hardest (R = 50 scored 1.697; R = 5 scored 0.773).

More lag columns are not a useful next change. Older history is already in the cumulative sum. The following trial changed the model instead of the feature list.

Log: `experiments/longer_history.json`. Script: `experiment_longer_history.py`.

## Trial 4 — forward LSTM, 10 epochs

One forward LSTM layer, hidden size 128, batch 512, learning rate 0.001, trained on the GPU. One example is one breath of 80 steps, read in time order. Inputs are the raw columns only: `u_in`, `u_out`, `time_step`, `R`, and `C`, standardized with the training-breath mean and standard deviation. Loss is L1 on inspiratory steps. The network does not look ahead. Lag columns were left out so the comparison is against a model that has to read the series itself.

The epoch count was fixed at 10 before the run. Validation MAE at epoch 10 was 1.583, which is 0.316 worse than trial 3.

| Epoch | Train MAE | Validation MAE |
| --- | ---: | ---: |
| 1 | 10.050 | 6.835 |
| 2 | 5.777 | 4.238 |
| 3 | 3.350 | 2.727 |
| 4 | 2.484 | 2.499 |
| 5 | 2.205 | 2.088 |
| 6 | 2.009 | 1.987 |
| 7 | 1.892 | 1.780 |
| 8 | 1.733 | 1.755 |
| 9 | 1.645 | 1.721 |
| 10 | 1.549 | 1.583 |

Train and validation error were still falling together at the end, so 10 epochs did not finish the comparison. The weights are in `experiments/forward_lstm.pt`.

Log: `experiments/forward_lstm.json`. Script: `experiment_forward_lstm.py`.

## Trial 5 — forward LSTM continued to epoch 200

Same network, split, and raw-column inputs as trial 4. Training resumed from the epoch-10 weights. The Adam moments started fresh, because trial 4 did not save them. The cap was epoch 200. The best validation weights were kept. If the validation MAE did not improve for 5 epochs, the learning rate was halved, down to 0.00001. If it did not improve for 10 epochs, the run would stop.

It ran all 200 epochs and did not stop early. The best validation MAE is the last epoch, **0.901**, after 390.7 seconds. That is 0.366 below the trial 3 tree. The score first beat the tree at epoch 21 (1.220).

| Epoch | Train MAE | Validation MAE | Learning rate |
| --- | ---: | ---: | ---: |
| 20 | 1.253 | 1.314 | 0.001 |
| 40 | 1.055 | 1.104 | 0.001 |
| 80 | 0.922 | 0.941 | 0.00025 |
| 100 | 0.895 | 0.915 | 0.0000625 |
| 137 | 0.882 | 0.906 | 0.00001 |
| 200 | 0.877 | 0.901 | 0.00001 |

The learning rate was cut at epochs 44, 63, 82, 90, 110, 125, and 137, then sat at the floor. After that the validation MAE only moved from 0.906 to 0.901. Train and validation stayed close, so the long run did not wander off into memorizing the training breaths. The stall rule never fired because a new best still appeared inside every 10-epoch window, even when the gain was only a few thousandths.

Log: `experiments/forward_lstm_200.json`. Weights: `experiments/forward_lstm_200.pt`. Script: `experiment_forward_lstm_continue.py`.

## Trial 6 — bidirectional LSTM

**Flag for the final report.** The backward pass is an outside modeling idea. Trials 1–5 did not discover it. The assistant suggested it because bidirectional recurrent models are a common approach on this challenge, and because each data file already contains the whole breath. Credit that in the final report. Do not describe the backward pass as a result of the earlier trials.

The network was trained from a new initialization. Trial 5’s weights were not loaded, because a backward pass adds parameters that those weights do not have. Otherwise the setup matches the forward LSTM: one layer, hidden size 128 in each direction, raw columns only, the same breath split, L1 loss on inspiratory steps, and a cap of 200 epochs. The learning rate was halved after 5 epochs without improvement. Training would stop after 10 epochs without improvement.

It ran all 200 epochs in 767.8 seconds and did not stop early. The best validation MAE is **0.768** at epoch 199. Epoch 200 was 0.769. That is 0.134 below the forward LSTM’s 0.901. The score first beat the forward model at epoch 80, and first went under 0.85 at epoch 100. Training MAE at epoch 199 was 0.739, still close to validation.

| Epoch | Train MAE | Validation MAE | Learning rate |
| --- | ---: | ---: | ---: |
| 1 | 8.407 | 4.622 | 0.001 |
| 20 | 1.226 | 1.223 | 0.001 |
| 50 | 0.992 | 1.009 | 0.0005 |
| 100 | 0.821 | 0.848 | 0.00025 |
| 150 | 0.770 | 0.791 | 0.000125 |
| 199 | 0.739 | 0.768 | 0.000015625 |

The learning rate was cut at epochs 50, 72, 106, 159, 172, and 195. It did not reach the floor of 0.00001. The error was still inching down at the end.

Log: `experiments/bilstm.json`. Weights: `experiments/bilstm.pt`. Script: `experiment_bilstm.py`.

## Trial 7 — two-layer bidirectional LSTM

The only change from trial 6 is a second bidirectional layer. Hidden size stays 128 per direction. Inputs stay the raw columns. The split, the L1 loss, and the 200-epoch rules are unchanged. Trial 6’s weights were not loaded. The backward pass is still the outside idea flagged below. The extra layer comes from trial 6’s own curve: training and validation error were close, and both were still high, so the network had settled for its size.

It ran all 200 epochs in 1622.9 seconds and did not stop early. The best validation MAE is **0.552** at epoch 195. Epoch 200 was the same to three digits. That is 0.216 below the one-layer network. The score first beat trial 6 at epoch 45. Training MAE at epoch 195 was 0.516.

| Epoch | Train MAE | Validation MAE | Learning rate |
| --- | ---: | ---: | ---: |
| 1 | 7.947 | 4.477 | 0.001 |
| 20 | 1.047 | 1.029 | 0.001 |
| 50 | 0.746 | 0.811 | 0.0005 |
| 100 | 0.586 | 0.613 | 0.00025 |
| 150 | 0.530 | 0.564 | 0.00003125 |
| 195 | 0.516 | 0.552 | 0.00001 |

The learning rate reached the floor of 0.00001 at epoch 177. From epoch 150 to 195 the validation MAE only moved from 0.564 to 0.552. Train and validation stayed within about 0.04, so the deeper network did not memorize the training breaths. The second layer helped, and this size has now leveled off too.

Log: `experiments/bilstm_2layer.json`. Weights: `experiments/bilstm_2layer.pt`. Script: `experiment_bilstm_2layer.py`.

## Trial 8 — four layers, hidden size 256

Depth and width were doubled together, from trial 7’s 2 layers at hidden size 128 to 4 bidirectional layers at hidden size 256. Trial 7’s weights were not loaded. Features, the breath split, the loss, the batch size, and the 200-epoch rules stayed the same. The backward pass is still the outside idea flagged below. Because both changes happened in one run, the score does not say whether depth or width did the work.

It ran all 200 epochs in 6813.8 seconds, about 1 hour 54 minutes, and did not stop early. The best validation MAE is **0.369** at epoch 197. Epoch 200 was 0.369. That is 0.183 below trial 7. The score first beat trial 7 at epoch 43, and first went under 0.40 at epoch 96. Training MAE at epoch 197 was 0.279, about 0.09 below validation, a wider gap than trial 7.

| Epoch | Train MAE | Validation MAE | Learning rate |
| --- | ---: | ---: | ---: |
| 1 | 6.974 | 4.555 | 0.001 |
| 20 | 0.741 | 0.967 | 0.001 |
| 50 | 0.501 | 0.554 | 0.0005 |
| 100 | 0.325 | 0.395 | 0.0000625 |
| 150 | 0.289 | 0.374 | 0.000015625 |
| 197 | 0.279 | 0.369 | 0.00001 |

The learning rate reached the floor of 0.00001 at epoch 193. From epoch 150 to 197 the validation MAE only moved from 0.374 to 0.369.

Log: `experiments/bilstm_4x256.json`. Weights: `experiments/bilstm_4x256.pt`. Script: `experiment_bilstm_4x256.py`.

## Trial 9 — engineered features and cosine decay

The network is the same as trial 8: 4 bidirectional layers, hidden size 256, batch 512, the same breath split, and inspiratory L1 loss. Trial 8’s weights were not loaded. Two things changed together, so the score does not say which one did the work. The learning rate followed a cosine from 0.001 down to 0.00001 across all 200 epochs, instead of the plateau cuts. Six features were added inside each breath: the cumulative sum of `u_in`, the one-step change in `u_in`, the cumulative sum of `u_in` only while `u_out` is 0, the steps remaining until `u_out` flips to 1, `u_in` times `R`, and the cumulative sum of `u_in` divided by `C`. The last two are the flagged pair below. The first four come from our own trials.

It ran all 200 epochs in 6735.8 seconds, about 1 hour 52 minutes. The best validation MAE is **0.244** at epoch 197. Epoch 200 was 0.245. That is 0.124 below trial 8. The score first beat trial 8 at epoch 77, first went under 0.35 at epoch 82, and first went under 0.25 at epoch 165. Training MAE at epoch 197 was 0.160, about 0.08 below validation.

| Epoch | Train MAE | Validation MAE | Learning rate |
| --- | ---: | ---: | ---: |
| 1 | 6.781 | 3.518 | 0.001 |
| 20 | 0.742 | 0.827 | 0.000978 |
| 50 | 0.427 | 0.478 | 0.000860 |
| 100 | 0.282 | 0.335 | 0.000513 |
| 150 | 0.187 | 0.256 | 0.000161 |
| 197 | 0.160 | 0.244 | 0.000011 |

Log: `experiments/bilstm_4x256_features.json`. Weights: `experiments/bilstm_4x256_features.pt`. Script: `experiment_bilstm_4x256_features.py`.

## Trial 10 — resume trial 9, upweight R = 50

Training started from the trial 9 weights. The network, features, split, and batch are unchanged. Inspiratory rows with `R = 50` count double in the training loss. The recorded score is still the ordinary unweighted inspiratory MAE. The cosine starts at 0.0003 and falls to 0.00001 over 200 epochs. Trial 9 started at 0.001.

The first epochs got worse. Epoch 1 was 0.257. The score first beat trial 9 at epoch 65. It ran all 200 epochs in 6722.2 seconds, about 1 hour 52 minutes. The best validation MAE is **0.222** at epoch 196. Epoch 200 was 0.223. That is 0.022 below trial 9. The `R = 50` slice moved from 0.318 in trial 9 to 0.283 at epoch 196. Training MAE at epoch 196 was 0.118, about 0.10 below validation, a wider gap than trial 9.

| Epoch | Train MAE | Validation MAE | R = 50 MAE | Learning rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.203 | 0.257 | 0.337 | 0.0003 |
| 50 | 0.178 | 0.252 | 0.325 | 0.000259 |
| 100 | 0.144 | 0.230 | 0.294 | 0.000157 |
| 150 | 0.125 | 0.225 | 0.286 | 0.000054 |
| 196 | 0.118 | 0.222 | 0.283 | 0.000010 |

Log: `experiments/bilstm_4x256_r50.json`. Weights: `experiments/bilstm_4x256_r50.pt`. Script: `experiment_bilstm_4x256_r50.py`.

## Ensemble check — median of trials 8, 9, and 10

No new training. Each saved model predicted the holdout breaths, and each row used the middle of the three pressures. The median MAE is **0.231**. The average of trials 9 and 10 alone is **0.228**. Both are worse than trial 10 by itself at 0.222. The `R = 50` slice of the median is 0.298, worse than trial 10’s 0.283. The three models miss the same hard breaths, and the weaker ones pull the middle value off trial 10’s prediction.

Script: `check_median_ensemble.py`.

## Trial 11 — R = 50 specialist, stitched with trial 10

Training started from the trial 10 weights and used only training breaths with `R = 50` (21,677 breaths). The loss on those breaths is ordinary inspiratory L1. There is no extra double-count, because the set is already `R = 50`. The cosine starts at 0.0001 and falls to 0.00001 over 200 epochs. Trial 10’s weights were not overwritten.

The recorded score stitches the two models. Holdout rows with `R = 50` use this specialist. Every other row keeps the trial 10 prediction. Before epoch 1, that stitch with the unchanged trial 10 weights scored 0.2222, the same number as trial 10.

It ran all 200 epochs in 5297.2 seconds, about 1 hour 28 minutes. The best stitched validation MAE is **0.220** at epoch 169. Epoch 200 was 0.220. That is 0.002 below trial 10. The score first beat trial 10 at epoch 14. The `R = 50` slice moved from 0.283 in trial 10 to 0.277 at epoch 169. Reaching an overall 0.20 would need that slice near 0.227 if `R = 5` and `R = 20` stayed where trial 10 left them. Training MAE on the `R = 50` breaths at epoch 169 was 0.112, about 0.165 below the `R = 50` validation slice.

| Epoch | Train MAE, R = 50 | Stitched MAE | R = 50 MAE | Learning rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.145 | 0.223 | 0.284 | 0.0001 |
| 50 | 0.128 | 0.222 | 0.282 | 0.000087 |
| 100 | 0.118 | 0.221 | 0.279 | 0.000056 |
| 150 | 0.113 | 0.220 | 0.277 | 0.000024 |
| 169 | 0.112 | 0.220 | 0.277 | 0.000016 |

Log: `experiments/bilstm_r50_specialist.json`. Weights: `experiments/bilstm_r50_specialist.pt`. Script: `experiment_bilstm_r50_specialist.py`.

## Model B — new seed, trial 9 schedule, then the R = 50 upweight

Same network, features, split, and batch as trial 9. The weight initialization and the row order use seed 7 instead of seed 42. Phase 1 is a fresh model with the trial 9 cosine, from 0.001 down to 0.00001 over 200 epochs. Its best validation MAE is 0.268 at epoch 199, in 6623.3 seconds. Trial 9 reached 0.244, so this seed was behind before the second phase started.

Phase 2 loads those best weights. Inspiratory rows with `R = 50` count double, and a fresh Adam follows a cosine from 0.0003 down to 0.00001 for 200 epochs. The recorded score is the unweighted inspiratory MAE. Both phases together took 13286.4 seconds, about 3 hours 41 minutes. The best validation MAE is **0.250** at epoch 198 of phase 2. The `R = 50` slice at that epoch is 0.320. Trial 10, from the stronger seed, reached 0.222 with the same second phase.

| Epoch | Train MAE | Validation MAE | R = 50 MAE | Learning rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.204 | 0.288 | 0.375 | 0.0003 |
| 50 | 0.179 | 0.275 | 0.349 | 0.000259 |
| 100 | 0.148 | 0.262 | 0.333 | 0.000157 |
| 150 | 0.123 | 0.252 | 0.323 | 0.000054 |
| 198 | 0.115 | 0.250 | 0.320 | 0.000010 |

Log: `experiments/model_b.json`. Weights: `experiments/model_b.pt`. Phase 1 weights: `experiments/model_b_phase1.pt`. Script: `experiment_model_b.py`.

## Model C — 4× loss on the hard cells, from scratch

Same network and trial 9 cosine as phase 1 of model B, with seed 11. Inspiratory rows with `R = 50` and `C = 20` or `C = 50` count four times in the training loss. Every other inspiratory row counts once. The recorded score is unweighted.

It ran 200 epochs in 6599.2 seconds, about 1 hour 50 minutes. The best validation MAE is **0.283** at epoch 199. The hard cells at that epoch score 0.374, which is worse than trial 11’s 0.312 on the same rows. Training MAE at epoch 199 was 0.181.

| Epoch | Train MAE | Validation MAE | Hard-cell MAE | Learning rate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6.735 | 4.084 | 5.143 | 0.001 |
| 50 | 0.511 | 0.555 | 0.652 | 0.000860 |
| 100 | 0.313 | 0.374 | 0.467 | 0.000513 |
| 150 | 0.212 | 0.299 | 0.388 | 0.000161 |
| 199 | 0.181 | 0.283 | 0.374 | 0.000010 |

Log: `experiments/model_c.json`. Weights: `experiments/model_c.pt`. Script: `experiment_model_c.py`.

## Average of trial 11 and model B

Model C started as soon as model B finished. On the hard cells, the residuals of trial 11 and model B correlate at 0.69, and trial 11 and model C correlate at 0.71. The misses overlap.

The median of trial 11, model B, and model C scores **0.221**. That is worse than trial 11 alone, because model C pulls the middle value off trial 11. With only trial 11 and model B, the middle of the two predictions is their average, and that average scores **0.218**. The hard cells in the average score 0.314, a bit worse than trial 11’s 0.312, so the 0.002 gain is on the other rows. This is the best holdout score so far. It is still above 0.20.

Script: `check_median_b_c.py`. Log: `experiments/median_b_c.json`.

## Flags for the final report

**Backward pass (trial 6 onward, including models B and C) — outside idea.** Credit it as a common approach on this challenge, suggested by the assistant after trial 5. The measured scores are our own holdout results. The choice to look backward cannot be described as a result of trials 1–5.

**`u_in * R` and cumulative `u_in / C` (trial 9 onward, including models B and C) — outside idea.** Credit this pair as a common constructed feature on this challenge. The column definitions support it, and the assistant suggested using it so the long run would include it. The other four added columns come from our own tree trials and from the finding that scored rows are the part of the breath where `u_out` is 0. Trial 10’s new change, doubling the loss on `R = 50`, trial 11’s specialist, and model C’s 4× loss on `R = 50` with `C = 20` or `C = 50`, come from our own holdout breakdown. Keep both flags in the final report.
