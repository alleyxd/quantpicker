# quantpicker

Which quant of a model should you run? Q4_K_M is the usual answer, but it's a guess. quantpicker measures it on
your own GPU: it downloads each GGUF quant, times it with `llama-bench`, compares its output to Q8_0 with
llama.cpp's KL-divergence test, and tells you the fastest one that's still close to the original.

```
quantpicker unsloth/Qwen3-8B-GGUF
```

```
┌──────────┬─────────┬───────────┬──────────────┬──────────┬─────────┬──────────┬────────┬─────────────────┐
│ quant    ┆ size    ┆ gen tok/s ┆ prompt tok/s ┆ mean KLD ┆ 99% KLD ┆ same top ┆ PPL    ┆ verdict         │
╞══════════╪═════════╪═══════════╪══════════════╪══════════╪═════════╪══════════╪════════╪═════════════════╡
│ Q2_K     ┆ 3.1 GiB ┆ 314       ┆ 10071        ┆ 0.2622   ┆ 2.860   ┆ 80.1%    ┆ +13.6% ┆ noticeable loss │
│ Q3_K_M   ┆ 3.8 GiB ┆ 270       ┆ 12148        ┆ 0.0946   ┆ 1.107   ┆ 88.0%    ┆ +5.3%  ┆ good            │
│ ★ IQ4_XS ┆ 4.3 GiB ┆ 264       ┆ 14326        ┆ 0.0371   ┆ 0.425   ┆ 92.2%    ┆ +2.9%  ┆ good            │
│ Q4_K_M   ┆ 4.7 GiB ┆ 237       ┆ 13085        ┆ 0.0302   ┆ 0.366   ┆ 93.0%    ┆ +1.3%  ┆ good            │
│ Q5_K_M   ┆ 5.4 GiB ┆ 213       ┆ 12813        ┆ 0.0131   ┆ 0.140   ┆ 95.3%    ┆ +1.8%  ┆ very good       │
│ Q6_K     ┆ 6.3 GiB ┆ 191       ┆ 11124        ┆ 0.0057   ┆ 0.055   ┆ 96.5%    ┆ +1.1%  ┆ near lossless   │
│ Q8_0     ┆ 8.1 GiB ┆ 162       ┆ 13833        ┆ 0        ┆ 0       ┆ 100%     ┆ -      ┆ reference       │
└──────────┴─────────┴───────────┴──────────────┴──────────┴─────────┴──────────┴────────┴─────────────────┘

Pick IQ4_XS: 4.3 GiB at 264 tok/s, mean KLD 0.0371 (good). The fastest quant within --max-kld 0.05.
```

That's Qwen3-8B on an RTX 5090. Q4_K_M is what most people would pick, but IQ4_XS is 11% faster and almost as
close to Q8_0. On your GPU the answer may be different, which is the point.

One command, nothing to set up. It finds your GPU, downloads the matching llama.cpp build (CUDA, Vulkan, Metal or
CPU), fetches the wikitext-2 test text, and deletes each model again after measuring it, so you only need disk space
for two quants at a time.

## Install

```
pip install quantpicker
```

or run it without installing: `uvx quantpicker unsloth/Qwen3-8B-GGUF`. There are also binaries for Windows, Linux
and macOS on the [releases](https://github.com/alleyxd/quantpicker/releases) page, or with Rust: `cargo install quantpicker`.

## Finding a model

Don't know the repo name? Search for it, then check what fits before downloading anything:

```
quantpicker search qwen3 8b              # GGUF repos on Hugging Face, best matches first
quantpicker list unsloth/Qwen3-8B-GGUF   # every quant, its size, and whether it fits your GPU
quantpicker unsloth/Qwen3-8B-GGUF        # measure and pick
```

Pass a repo without GGUF files, like `Qwen/Qwen3-8B`, and it suggests GGUF versions of it instead.

## What the numbers mean

- **gen tok/s / prompt tok/s**: `llama-bench` generation and prompt-processing speed, all layers on the GPU.
- **mean KLD**: how far the quant's next-token distribution is from the reference, averaged over every token of the
  test text. This is the number llama.cpp developers use to compare quants. Lower is better, 0 is identical.
- **99% KLD**: the worst 1% of tokens. A quant can look fine on average and still go badly wrong on hard tokens.
- **same top**: how often the quant picks the same most likely token as the reference.
- **PPL**: perplexity change against the reference. Easy to read, but KLD catches damage that perplexity hides.

The pick is the fastest quant that fits in memory with mean KLD under `--max-kld` (default 0.05). The verdicts are
a rough guide: under 0.01 is near lossless, under 0.1 is good, above 0.3 you'll notice.

## Options

```
quantpicker search <words>      find GGUF repos
quantpicker list <repo>         show quants and what fits, no download
quantpicker <repo> [options]    measure and pick

  --quants Q4_K_M,Q5_K_M   test these (or "all"); default Q2_K, Q3_K_M, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K
  --reference Q8_0         what to compare against (falls back to BF16 or F16)
  --max-kld 0.05           quality you're willing to give up
  --vram 24                plan for this much memory (GiB) instead of what's free now
  --headroom 1.5           GiB to leave for context and buffers
  --chunks 40              test-text chunks for the quality check; more is slower and tighter
  --text file.txt          measure quality on your own text instead of wikitext-2
  --llama-dir path         use your own llama.cpp build
  --backend vulkan         force a backend: cuda, vulkan, metal, cpu
  --keep                   keep the downloaded models
  --json out.json          save every number
```

Gated repos work with `HF_TOKEN` set. To download through a mirror, set `HF_ENDPOINT` (for example
`https://hf-mirror.com`), the same variable the Hugging Face tools use.

## Notes

- KL divergence is measured on general English text. If you care about code or another language, pass your own text
  with `--text`; quants don't lose quality evenly across domains.
- Quality numbers are relative to the reference. With the Q8_0 default, a quant can't score better than Q8_0.
- Speed is measured with the whole model on the GPU. Quants that don't fit are listed but not run.
- Only NVIDIA (`nvidia-smi`) and Apple GPUs are detected automatically. For anything else, pass `--vram`.

## License

MIT
