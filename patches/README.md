# Community patches 🧩

Small weight edits anyone can apply in a second. One folder per base model:

```
patches/
  qwen2.5-0.5b-instruct/
    no-looping.safetensors
    no-looping.md          # what it fixes, how it was found, before/after numbers
```

Apply:
```bash
myai patch info  patches/qwen2.5-0.5b-instruct/no-looping.safetensors
myai patch apply Qwen/Qwen2.5-0.5B-Instruct patches/qwen2.5-0.5b-instruct/no-looping.safetensors --out my-model
```

To share a patch, open a PR with:
1. the `.safetensors` file (made with `myai patch auto`, a `Recorder`, or `myai patch diff`)
2. a short `.md` with what it fixes, which tests you used, and **held-out** before/after scores
3. what it might break (every edit has side effects, so say what you checked)

Patches contain a checksum of the weights they were made for, so they refuse to apply to the wrong model.
