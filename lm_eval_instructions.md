

(lingua_241023) ya255@abdelfattah-compute-03:/scratch/ya255/lm-evaluation-harness$ git rev-parse HEAD
fb2e4b593c9fe19ae0b127bce30b4494bd683368

Above is the lm-evaluation-harness commit ID used, apply the patch from this commit ID as

```
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout fb2e4b593c9fe19ae0b127bce30b4494bd683368
# Copy patch file to the lm-evaluation-harness
cd lm-evaluation-harness
git apply lm_eval_changes.patch
```