# Average of trial 11 and model B

Each test row is the average of the trial 11 stitch and model B. The trial 11 stitch uses the R=50 specialist on `R = 50` and the trial 10 model on every other row. Model C is not included. This is the best holdout score so far.

The holdout was scored again before the file was written. Inspiratory MAE was **0.2178**. Trial 11 alone is 0.2199, in `submission_trial11`.

| Check | Result |
| --- | --- |
| Rows | 603,600, one per test id |
| Id order | Matches `data/sample_submission.csv` |
| Pressures | All finite, from -1.331 to 64.838 |

Submit from the `ventilator-pressure-prediction` directory:

```powershell
aicodinggym mle submit ventilator-pressure-prediction -F submission_11_b/submission.csv -m "average of trial 11 stitch and model B"
```

## Flags to keep in the final report

The backward pass is an outside idea, suggested after trial 5 because bidirectional models are a common approach on this challenge.

`u_in * R` and the cumulative sum of `u_in` divided by `C` are an outside idea as well. Model B uses both. Its new seed, and the choice to average it with trial 11, come from our own holdout results.

## Holdout error by lung setting

| Group | Inspiratory MAE |
| --- | ---: |
| R = 5 | 0.155 |
| R = 20 | 0.204 |
| R = 50 | 0.277 |
| C = 10 | 0.219 |
| C = 20 | 0.209 |
| C = 50 | 0.224 |

`submission_check.json` has the exact figures.
