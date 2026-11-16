# Trial 9 submission

Prediction file for the AI Coding Gym ventilator challenge, from the trial 9 weights (epoch 197).

The holdout was scored again with these weights before the file was written. Inspiratory MAE was **0.2445**, the same number stored in the checkpoint.

| Check | Result |
| --- | --- |
| Rows | 603,600, one per test id |
| Id order | Matches `data/sample_submission.csv` |
| Pressures | All finite, from -1.676 to 64.850 |

Submit from the `ventilator-pressure-prediction` directory:

```powershell
aicodinggym mle submit ventilator-pressure-prediction -F submission_trial9/submission.csv -m "trial 9 four-layer bidirectional LSTM"
```

## Flags to keep in the final report

The backward pass is an outside idea, suggested after trial 5 because bidirectional models are a common approach on this challenge.

`u_in * R` and the cumulative sum of `u_in` divided by `C` are an outside idea as well. The column definitions support them, and they are a common constructed feature on this challenge. The other added columns come from our own trials.

## Holdout error by lung setting

| Group | Inspiratory MAE |
| --- | ---: |
| R = 5 | 0.169 |
| R = 20 | 0.225 |
| R = 50 | 0.318 |
| C = 10 | 0.244 |
| C = 20 | 0.239 |
| C = 50 | 0.250 |

`submission_check.json` has the exact figures.
