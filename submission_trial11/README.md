# Trial 11 submission

Prediction file from the trial 11 stitch (specialist epoch 169). Rows with `R = 50` use the specialist. Every other row keeps the trial 10 prediction. The average with model B in `submission_11_b` is the better file, at 0.2178. Keep this one as the backup.

The holdout was scored again with these weights before the file was written. Inspiratory MAE was **0.2199**, the same number stored in the specialist checkpoint. The trial 10 file in `submission_trial10` is the earlier backup, at 0.2222.

| Check | Result |
| --- | --- |
| Rows | 603,600, one per test id |
| Id order | Matches `data/sample_submission.csv` |
| Pressures | All finite, from -1.221 to 64.842 |

Submit from the `ventilator-pressure-prediction` directory:

```powershell
aicodinggym mle submit ventilator-pressure-prediction -F submission_trial11/submission.csv -m "trial 11 R=50 specialist stitched with trial 10"
```

## Flags to keep in the final report

The backward pass is an outside idea, suggested after trial 5 because bidirectional models are a common approach on this challenge.

`u_in * R` and the cumulative sum of `u_in` divided by `C` are an outside idea as well. Training a specialist only on `R = 50` and stitching it back comes from our own holdout breakdown.

## Holdout error by lung setting

| Group | Inspiratory MAE |
| --- | ---: |
| R = 5 | 0.159 |
| R = 20 | 0.209 |
| R = 50 | 0.277 |
| C = 10 | 0.218 |
| C = 20 | 0.217 |
| C = 50 | 0.225 |

`submission_check.json` has the exact figures.
