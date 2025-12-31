# MIPROv2 Compile() Method Analysis

## Summary

For `auto="medium"` mode in DSpy's MIPROv2 optimizer, the `compile()` method performs **25 trials** (epochs).

## Key Findings

### 1. Number of Trials

From `AUTO_RUN_SETTINGS` in `mipro_optimizer_v2.py`:
```python
AUTO_RUN_SETTINGS = {
    "light": {"num_trials": 7, "val_size": 100},
    "medium": {"num_trials": 25, "val_size": 300},
    "heavy": {"num_trials": 50, "val_size": 1000},
}
```

**For `auto="medium"`: `num_trials = 25`**

### 2. How Trials Are Executed

In `_optimize_prompt_parameters()` method:
- **1 default/baseline trial** (trial number -1): Evaluates the original program
- **24 optimization trials** (trial numbers 1-24): Uses Optuna's Bayesian optimization
  ```python
  study.optimize(objective, n_trials=num_trials-1)  # n_trials=24
  ```
- **Total: 25 trials**

### 3. Number of Unique Prompts

Each trial generates a **unique prompt combination** consisting of:
- **One instruction** selected from instruction candidates
- **Few-shot demos** selected from demo candidates

The number of instruction candidates is determined by:
```python
num_candidates = int(np.round(np.min([num_trials * num_vars, (1.5 * num_trials) / num_vars])))
```

For a `ChainOfThought` module with 1 predictor:
- `num_vars = 1 * 2 = 2` (accounts for instruction + few-shot demo variables)
- For `num_trials = 25`: `num_candidates = min(50, 18.75) ≈ 19` instruction candidates

**However, the optimizer only runs 25 trials**, each selecting a unique combination from these candidates. Therefore:

**The number of unique prompts ≈ 25 (one per trial)**

This matches the user's expectation that "the unique prompt numbers should be very close to the epoch/num_trials."

### 4. Code Flow

1. `optimizer.compile(module, trainset=dataset.train, **kwargs)` is called
2. `compile()` calls `_set_hyperparams_from_run_mode()` which sets `num_trials = 25` for "medium"
3. `compile()` calls `_propose_instructions()` which generates ~19 instruction candidates
4. `compile()` calls `_optimize_prompt_parameters()` which:
   - Runs 1 default trial (baseline)
   - Runs 24 optimization trials via Optuna
   - Each trial creates a unique prompt combination
5. Total: **25 unique prompts** (one per trial)

## Conclusion

For `auto="medium"` mode:
- **Number of trials/epochs: 25**
- **Number of unique prompts: ~25** (one unique prompt combination per trial)


