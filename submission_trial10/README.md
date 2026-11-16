# Trial 10 submission

Prediction file from the trial 10 weights (epoch 196). Trial 11 in `submission_trial11` is the better file, at 0.2199. Keep this one as the backup.

The holdout was scored again with these weights before the file was written. Inspiratory MAE was **0.2222**, the same number stored in the checkpoint. The trial 9 file in `submission_trial9` is the earlier backup, at 0.2445.

| Check | Result |
| --- | --- |
| Rows | 603,600, one per test id |
| Id order | Matches `data/sample_submission.csv` |
| Pressures | All finite, from -1.253 to 64.837 |

Submit from the `ventilator-pressure-prediction` directory:

```powershell
aicodinggym mle submit ventilator-pressure-prediction -F submission_trial10/submission.csv -m "trial 10 R=50 upweighted bidirectional LSTM"
```

## Flags to keep in the final report

The backward pass is an outside idea, suggested after trial 5 because bidirectional models are a common approach on this challenge.

`u_in * R` and the cumulative sum of `u_in` divided by `C` are an outside idea as well. Doubling the loss on `R = 50` comes from our own holdout breakdown.

## Holdout error by lung setting

| Group | Inspiratory MAE |
| --- | ---: |
| R = 5 | 0.159 |
| R = 20 | 0.209 |
| R = 50 | 0.282 |
| C = 10 | 0.222 |
| C = 20 | 0.219 |
| C = 50 | 0.226 |

`submission_check.json` has the exact figures.
