mod gguf;
mod gpu;
mod hf;
mod llama;
mod pick;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{CommandFactory, Parser, Subcommand};

use gpu::Gpu;
use hf::Quant;
use llama::Eval;
use pick::{GIB, Row, gb};

const DEFAULT_QUANTS: &[&str] = &[
    "Q2_K", "Q3_K_M", "IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0",
];
const WIKITEXT: &str =
    "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip";

const EXAMPLES: &str = "Examples:
  quantpicker search qwen3 8b             find GGUF repos on Hugging Face
  quantpicker list unsloth/Qwen3-8B-GGUF  see its quants and which fit your GPU
  quantpicker unsloth/Qwen3-8B-GGUF       measure them and get a pick";

/// Measures which GGUF quant of a model is worth running on your GPU: real speed and real quality loss,
/// not estimates.
#[derive(Parser)]
#[command(
    version,
    args_conflicts_with_subcommands = true,
    arg_required_else_help = true,
    after_help = EXAMPLES
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,

    #[command(flatten)]
    run: Args,
}

#[derive(Subcommand)]
enum Cmd {
    /// Find GGUF repos on Hugging Face, most downloaded first
    Search {
        /// What to look for, e.g. "qwen3 8b" or "gemma"
        #[arg(required = true)]
        query: Vec<String>,

        /// How many repos to show
        #[arg(long, default_value_t = 15)]
        limit: usize,
    },
    /// Show the quants in a repo and which fit your GPU, without downloading anything
    List {
        /// Hugging Face repo, e.g. unsloth/Qwen3-8B-GGUF
        repo: String,

        /// Memory to plan for, in GiB. Default: free memory on the GPU.
        #[arg(long)]
        vram: Option<f64>,

        /// Memory to leave free for context and buffers, in GiB.
        #[arg(long, default_value_t = 1.5)]
        headroom: f64,

        /// Plan for this many tokens of context: its KV cache is counted against GPU memory.
        #[arg(long)]
        context: Option<u64>,
    },
}

#[derive(clap::Args)]
struct Args {
    /// Hugging Face repo with GGUF files, e.g. unsloth/Qwen3-8B-GGUF
    repo: Option<String>,

    /// Quants to test, comma separated, or "all". Default: the common ones from Q2_K to Q8_0.
    #[arg(long, value_delimiter = ',')]
    quants: Vec<String>,

    /// Quant to compare against.
    #[arg(long, default_value = "Q8_0")]
    reference: String,

    /// Memory to plan for, in GiB. Default: free memory on the GPU.
    #[arg(long)]
    vram: Option<f64>,

    /// Memory to leave free for context and buffers, in GiB.
    #[arg(long, default_value_t = 1.5)]
    headroom: f64,

    /// Plan for this many tokens of context: its KV cache is counted against GPU memory.
    #[arg(long)]
    context: Option<u64>,

    /// Highest mean KL divergence you'll accept. Lower means closer to the reference.
    #[arg(long, default_value_t = 0.05)]
    max_kld: f64,

    /// Text chunks used to measure quality. More is slower and more precise.
    #[arg(long, default_value_t = 40)]
    chunks: u32,

    /// Context size per chunk.
    #[arg(long, default_value_t = 512)]
    ctx: u32,

    /// Text to measure quality on. Default: wikitext-2 test set.
    #[arg(long)]
    text: Option<PathBuf>,

    /// llama-bench repetitions per quant.
    #[arg(long, default_value_t = 3)]
    reps: u32,

    /// Use the llama.cpp binaries in this folder instead of downloading them.
    #[arg(long)]
    llama_dir: Option<PathBuf>,

    /// llama.cpp backend to download: cuda, vulkan, metal or cpu.
    #[arg(long)]
    backend: Option<String>,

    /// Where to download models. They are deleted after testing unless --keep is set.
    #[arg(long)]
    models_dir: Option<PathBuf>,

    /// Keep downloaded models.
    #[arg(long)]
    keep: bool,

    /// Write the results as JSON to this file.
    #[arg(long)]
    json: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();
    let token = std::env::var("HF_TOKEN").ok();
    let result = match cli.cmd {
        Some(Cmd::Search { query, limit }) => search(&query.join(" "), limit, token.as_deref()),
        Some(Cmd::List {
            repo,
            vram,
            headroom,
            context,
        }) => list(&repo, vram, headroom, context, token.as_deref()),
        None => match cli.run.repo.as_deref() {
            // "qwen3-8b" isn't a repo name, so treat it as a search
            Some(r) if !r.contains('/') => {
                eprintln!(
                    "\"{r}\" isn't a repo name (those look like owner/name). Searching instead:
"
                );
                search(r, 10, token.as_deref())
            }
            Some(_) => run(cli.run, token.as_deref()),
            None => Cli::command().print_help().map_err(Into::into),
        },
    };
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

/// Returns the GPU (if found), the memory budget and what's left of it after headroom.
fn budget(vram: Option<f64>, headroom: f64) -> Result<(Option<Gpu>, u64, u64)> {
    let gpu = gpu::detect();
    let budget = match (vram, &gpu) {
        (Some(v), _) => (v * GIB) as u64,
        (None, Some(g)) => g.free,
        (None, None) => {
            bail!("couldn't detect a GPU; pass --vram with the memory you want to plan for, in GiB")
        }
    };
    Ok((gpu, budget, budget.saturating_sub((headroom * GIB) as u64)))
}

fn search(query: &str, limit: usize, token: Option<&str>) -> Result<()> {
    let repos = hf::search(query, limit, token)?;
    if repos.is_empty() {
        println!(
            "No GGUF repos match \"{query}\". Try fewer or shorter words, e.g. just the model family."
        );
        return Ok(());
    }
    println!("{}", pick::repo_table(&repos));
    println!(
        "
Next: quantpicker list {}",
        repos[0].id
    );
    Ok(())
}

fn list(
    repo: &str,
    vram: Option<f64>,
    headroom: f64,
    context: Option<u64>,
    token: Option<&str>,
) -> Result<()> {
    let (gpu, budget, usable) = budget(vram, headroom)?;
    let quants = quants_or_suggest(repo, token)?;
    let usable = usable.saturating_sub(context_kv(repo, &quants, context, token)?);
    let where_ = match &gpu {
        Some(g) if vram.is_none() => format!("{} with {} free", g.name, gb(g.free)),
        _ => gb(budget),
    };
    println!(
        "{repo} on {where_} (keeping {headroom} GiB for context):
"
    );
    println!("{}", pick::quant_table(&quants, usable, DEFAULT_QUANTS));
    println!(
        "
Next: quantpicker {repo}    (measures the ones marked \"tested\")"
    );
    Ok(())
}

/// Lists a repo's quants, or fails with a few GGUF repos to try instead.
fn quants_or_suggest(repo: &str, token: Option<&str>) -> Result<Vec<Quant>> {
    let quants = hf::list_quants(repo, token)?;
    if !quants.is_empty() {
        return Ok(quants);
    }
    let alts = hf::search(hf::search_term(repo), 5, token).unwrap_or_default();
    if alts.is_empty() {
        bail!("{repo} has no GGUF files. Try: quantpicker search <model name>");
    }
    let names: Vec<String> = alts.iter().map(|r| format!("  {}", r.id)).collect();
    bail!(
        "{repo} has no GGUF files. These GGUF versions might be what you want:
{}",
        names.join(
            "
"
        )
    )
}

fn run(args: Args, token: Option<&str>) -> Result<()> {
    let repo = args.repo.clone().unwrap_or_default();
    let (gpu, budget, usable) = budget(args.vram, args.headroom)?;
    match &gpu {
        Some(g) => eprintln!("GPU: {} ({} free of {})", g.name, gb(g.free), gb(g.total)),
        None => eprintln!("GPU: not detected, planning for {}", gb(budget)),
    }

    let all = quants_or_suggest(&repo, token)?;
    let usable = usable.saturating_sub(context_kv(&repo, &all, args.context, token)?);
    let reference = pick_reference(&all, &args.reference)?;
    let mut todo = select(&all, &args.quants, &reference.name)?;
    todo.retain(|q| q.name != reference.name);
    eprintln!(
        "{}: {} quants, testing {} against {} ({})",
        repo,
        all.len(),
        todo.iter()
            .map(|q| q.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        reference.name,
        gb(reference.size)
    );
    if reference.size > usable {
        eprintln!(
            "  note: the reference doesn't fit in GPU memory, so measuring it will be slow or may fail"
        );
    }

    let llama = llama::find_or_install(
        args.llama_dir.as_deref(),
        args.backend.as_deref(),
        gpu.as_ref(),
    )?;
    eprintln!("llama.cpp: {} ({})", llama.version, llama.dir.display());

    let data = dirs::data_local_dir()
        .context("no data dir")?
        .join("quantpicker");
    let models = args
        .models_dir
        .clone()
        .unwrap_or_else(|| data.join("models").join(repo.replace('/', "--")));
    fs::create_dir_all(&models)?;
    let text = match &args.text {
        Some(t) => t.clone(),
        None => wikitext(&data)?,
    };
    let ev = Eval {
        text: &text,
        ctx: args.ctx,
        chunks: args.chunks,
    };
    let logits = models.join("reference.kld");

    eprintln!("\n[ref] {} downloading", reference.name);
    let ref_path = hf::download(&repo, &reference, &models, token)?;
    eprintln!("[ref] {} saving reference logits", reference.name);
    llama::save_reference(&llama, &ref_path, &ev, &logits)?;
    let speed = llama::bench(&llama, &ref_path, args.reps).ok();
    let mut rows = vec![Row {
        quant: reference.name.clone(),
        size: reference.size,
        fits: reference.size <= usable,
        reference: true,
        gen_tps: speed.as_ref().map(|s| s.generate),
        prompt_tps: speed.as_ref().map(|s| s.prompt),
        mean_kld: Some(0.0),
        p99_kld: Some(0.0),
        same_top: Some(100.0),
        ppl_ratio: Some(1.0),
        error: None,
    }];

    for (i, q) in todo.iter().enumerate() {
        let tag = format!("[{}/{}] {}", i + 1, todo.len(), q.name);
        let mut row = Row {
            quant: q.name.clone(),
            size: q.size,
            fits: q.size <= usable,
            reference: false,
            gen_tps: None,
            prompt_tps: None,
            mean_kld: None,
            p99_kld: None,
            same_top: None,
            ppl_ratio: None,
            error: None,
        };
        if !row.fits {
            eprintln!(
                "{tag} skipped, {} doesn't fit in {}",
                gb(q.size),
                gb(usable)
            );
            rows.push(row);
            continue;
        }
        eprintln!("{tag} downloading {}", gb(q.size));
        let result = hf::download(&repo, q, &models, token).and_then(|path| {
            eprintln!("{tag} measuring");
            let s = llama::bench(&llama, &path, args.reps)?;
            let k = llama::compare(&llama, &path, &ev, &logits)?;
            if !args.keep {
                remove(&models, q);
            }
            Ok((s, k))
        });
        match result {
            Ok((s, k)) => {
                eprintln!("{tag} {:.0} tok/s, mean KLD {:.4}", s.generate, k.mean_kld);
                row.gen_tps = Some(s.generate);
                row.prompt_tps = Some(s.prompt);
                row.mean_kld = Some(k.mean_kld);
                row.p99_kld = Some(k.p99_kld);
                row.same_top = Some(k.same_top);
                row.ppl_ratio = Some(k.ppl_ratio);
            }
            Err(e) => {
                eprintln!("{tag} failed: {e:#}");
                row.error = Some(format!("{e:#}"));
            }
        }
        rows.push(row);
    }

    let _ = fs::remove_file(&logits);
    if !args.keep {
        remove(&models, &reference);
    }

    rows.sort_by_key(|r| r.size);
    let best = pick::recommend(&rows, args.max_kld);
    println!("\n{}", pick::table(&rows, best));
    match best {
        Some(b) if b.mean_kld.unwrap_or(1.0) <= args.max_kld => println!(
            "\nPick {}: {} at {:.0} tok/s, mean KLD {:.4} ({}). The fastest quant within --max-kld {}.",
            b.quant,
            gb(b.size),
            b.gen_tps.unwrap_or(0.0),
            b.mean_kld.unwrap_or(0.0),
            pick::verdict(b.mean_kld.unwrap_or(0.0)),
            args.max_kld
        ),
        Some(b) => println!(
            "\nNothing that fits is within --max-kld {}. Closest is {} (mean KLD {:.4}).",
            args.max_kld,
            b.quant,
            b.mean_kld.unwrap_or(0.0)
        ),
        None => println!("\nNo quant could be measured."),
    }

    if let Some(path) = &args.json {
        let out = serde_json::json!({
            "repo": repo,
            "gpu": gpu.as_ref().map(|g| &g.name),
            "llama_cpp": llama.version,
            "reference": reference.name,
            "chunks": args.chunks,
            "ctx": args.ctx,
            "pick": best.map(|b| &b.quant),
            "results": rows,
        });
        fs::write(path, serde_json::to_string_pretty(&out)?)?;
    }
    Ok(())
}

/// KV cache bytes for `context` tokens, read from the model's GGUF header. Every quant of a model has the same
/// shape, so the smallest file is enough. 0 when no context was asked for or the architecture can't be sized.
fn context_kv(
    repo: &str,
    quants: &[Quant],
    context: Option<u64>,
    token: Option<&str>,
) -> Result<u64> {
    let Some(ctx) = context else { return Ok(0) };
    let Some(q) = quants.iter().min_by_key(|q| q.size) else {
        return Ok(0);
    };
    match hf::shape(repo, q, token)? {
        Some(shape) => {
            let kv = gguf::kv_bytes(&shape, ctx);
            eprintln!(
                "KV cache for {ctx} tokens ({}, {} layers, 16-bit): {}",
                shape.arch,
                shape.layers,
                gb(kv)
            );
            Ok(kv)
        }
        None => {
            eprintln!("can't size the KV cache for this model's attention; --context is ignored");
            Ok(0)
        }
    }
}

fn pick_reference(all: &[Quant], wanted: &str) -> Result<Quant> {
    if let Some(q) = all.iter().find(|q| q.name.eq_ignore_ascii_case(wanted)) {
        return Ok(q.clone());
    }
    // fall back to the most precise format the repo has
    ["Q8_0", "BF16", "F16", "Q6_K"]
        .iter()
        .find_map(|n| all.iter().find(|q| q.name == *n))
        .cloned()
        .with_context(|| {
            format!("no {wanted} in this repo to compare against; pick one with --reference")
        })
}

fn select(all: &[Quant], wanted: &[String], reference: &str) -> Result<Vec<Quant>> {
    if wanted.iter().any(|w| w.eq_ignore_ascii_case("all")) {
        return Ok(all
            .iter()
            .filter(|q| !matches!(q.name.as_str(), "BF16" | "F16" | "F32"))
            .cloned()
            .collect());
    }
    let names: Vec<String> = if wanted.is_empty() {
        DEFAULT_QUANTS.iter().map(|s| s.to_string()).collect()
    } else {
        wanted.iter().map(|w| w.to_uppercase()).collect()
    };
    // asking for a quant the repo doesn't have is almost always a typo, so say so instead of skipping it
    let missing: Vec<&String> = names
        .iter()
        .filter(|n| !wanted.is_empty() && !all.iter().any(|q| &q.name == *n))
        .collect();
    if !missing.is_empty() {
        let have: Vec<&str> = all.iter().map(|q| q.name.as_str()).collect();
        bail!(
            "not in this repo: {}. Available: {}",
            missing
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            have.join(", ")
        );
    }
    Ok(all
        .iter()
        .filter(|q| names.contains(&q.name) || q.name == reference)
        .cloned()
        .collect())
}

fn remove(dir: &Path, q: &Quant) {
    for f in &q.files {
        let _ = fs::remove_file(dir.join(f.rsplit('/').next().unwrap()));
    }
}

fn wikitext(data: &Path) -> Result<PathBuf> {
    let path = data.join("wiki.test.raw");
    if path.exists() {
        return Ok(path);
    }
    fs::create_dir_all(data)?;
    eprintln!("downloading wikitext-2 test set");
    let bytes = reqwest::blocking::get(WIKITEXT)?
        .error_for_status()?
        .bytes()?;
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
    let mut file = zip
        .by_name("wikitext-2-raw/wiki.test.raw")
        .context("wiki.test.raw missing from archive")?;
    std::io::copy(&mut file, &mut fs::File::create(&path)?)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quants(names: &[&str]) -> Vec<Quant> {
        names
            .iter()
            .map(|n| Quant {
                name: n.to_string(),
                files: vec![],
                size: 0,
            })
            .collect()
    }

    #[test]
    fn default_selection_skips_what_the_repo_lacks() {
        let all = quants(&["Q4_K_M", "Q8_0", "BF16"]);
        let picked = select(&all, &[], "Q8_0").unwrap();
        assert_eq!(
            picked.iter().map(|q| q.name.as_str()).collect::<Vec<_>>(),
            ["Q4_K_M", "Q8_0"]
        );
    }

    #[test]
    fn unknown_requested_quant_is_an_error() {
        let all = quants(&["Q4_K_M", "Q8_0"]);
        let err = select(&all, &["q4_k_m".into(), "Q5_K_S".into()], "Q8_0").unwrap_err();
        assert!(err.to_string().contains("not in this repo: Q5_K_S"));
        assert_eq!(select(&all, &["q4_k_m".into()], "Q8_0").unwrap().len(), 2);
    }
}
