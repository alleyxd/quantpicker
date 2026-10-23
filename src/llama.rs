use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Deserialize;

use crate::gpu::Gpu;

const EXE: &str = std::env::consts::EXE_SUFFIX;

pub struct Llama {
    pub dir: PathBuf,
    pub version: String,
}

impl Llama {
    fn bin(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{name}{EXE}"))
    }
}

/// Uses llama.cpp from `dir` or PATH if it's there, otherwise downloads the right release build.
pub fn find_or_install(
    dir: Option<&Path>,
    backend: Option<&str>,
    gpu: Option<&Gpu>,
) -> Result<Llama> {
    if let Some(d) = dir {
        return found(d)
            .with_context(|| format!("no llama-bench or llama-perplexity in {}", d.display()));
    }
    if let Some(l) = on_path() {
        return Ok(l);
    }
    install(backend, gpu)
}

fn found(dir: &Path) -> Option<Llama> {
    let ok = ["llama-bench", "llama-perplexity"]
        .iter()
        .all(|b| dir.join(format!("{b}{EXE}")).is_file());
    ok.then(|| Llama {
        dir: dir.to_path_buf(),
        version: "local".into(),
    })
}

fn on_path() -> Option<Llama> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|d| found(&d))
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

#[derive(Deserialize, Clone)]
struct Asset {
    name: String,
    browser_download_url: String,
}

fn install(backend: Option<&str>, gpu: Option<&Gpu>) -> Result<Llama> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("quantpicker")
        .timeout(None)
        .build()?;
    let releases: Vec<Release> = client
        .get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10")
        .send()?
        .error_for_status()?
        .json()?;
    // the newest entries with real build assets; some tags only carry a marker file
    let rel = releases
        .into_iter()
        .find(|r| r.assets.len() > 5)
        .context("no llama.cpp release with builds")?;
    let names: Vec<&str> = rel.assets.iter().map(|a| a.name.as_str()).collect();
    let picked = pick_assets(
        &names,
        std::env::consts::OS,
        std::env::consts::ARCH,
        backend,
        gpu.and_then(|g| g.cuda),
    )?;

    let root = dirs::data_local_dir()
        .context("no data dir")?
        .join("quantpicker")
        .join("llama.cpp");
    let dir = root.join(format!("{}-{}", rel.tag_name, picked.backend));
    if let Some(l) = find_bin_dir(&dir) {
        return Ok(Llama {
            version: rel.tag_name,
            ..l
        });
    }
    fs::create_dir_all(&dir)?;
    for name in &picked.files {
        let asset = rel.assets.iter().find(|a| &a.name == name).unwrap();
        eprintln!("  downloading llama.cpp {} ({})", rel.tag_name, asset.name);
        let archive = dir.join(&asset.name);
        let mut resp = client
            .get(&asset.browser_download_url)
            .send()?
            .error_for_status()?;
        resp.copy_to(&mut File::create(&archive)?)?;
        unpack(&archive, &dir)?;
        fs::remove_file(&archive)?;
    }
    let l = find_bin_dir(&dir).context("llama.cpp download didn't contain llama-bench")?;
    Ok(Llama {
        version: rel.tag_name,
        ..l
    })
}

fn find_bin_dir(dir: &Path) -> Option<Llama> {
    if !dir.is_dir() {
        return None;
    }
    if let Some(l) = found(dir) {
        return Some(l);
    }
    fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|e| e.path().is_dir())
        .find_map(|e| find_bin_dir(&e.path()))
}

fn unpack(archive: &Path, dest: &Path) -> Result<()> {
    let name = archive.to_string_lossy();
    if name.ends_with(".zip") {
        zip::ZipArchive::new(File::open(archive)?)?.extract(dest)?;
    } else {
        tar::Archive::new(flate2::read::GzDecoder::new(File::open(archive)?)).unpack(dest)?;
    }
    Ok(())
}

#[derive(Debug, PartialEq)]
pub struct Picked {
    pub backend: String,
    pub files: Vec<String>,
}

/// Chooses which release archives to download for this machine.
pub fn pick_assets(
    names: &[&str],
    os: &str,
    arch: &str,
    backend: Option<&str>,
    cuda: Option<(u32, u32)>,
) -> Result<Picked> {
    let os = match os {
        "windows" => "win",
        "linux" => "ubuntu",
        "macos" => "macos",
        other => bail!("no prebuilt llama.cpp for {other}; pass --llama-dir"),
    };
    let arch = match arch {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => bail!("no prebuilt llama.cpp for {other}; pass --llama-dir"),
    };
    let one = |backend: &str, re: &str| -> Option<Picked> {
        let re = Regex::new(re).unwrap();
        names.iter().find(|n| re.is_match(n)).map(|n| Picked {
            backend: backend.into(),
            files: vec![n.to_string()],
        })
    };
    let want = backend.unwrap_or(if os == "macos" {
        "metal"
    } else if cuda.is_some() {
        "cuda"
    } else {
        "vulkan"
    });
    let picked = match want {
        "metal" => one(
            "metal",
            &format!(r"^llama-[^-]+-bin-macos-{arch}\.tar\.gz$"),
        ),
        "cuda" => pick_cuda(names, os, arch, cuda.unwrap_or((99, 0))),
        "vulkan" => one(
            "vulkan",
            &format!(r"^llama-[^-]+-bin-{os}-vulkan-{arch}\.(zip|tar\.gz)$"),
        ),
        "cpu" if os == "win" => one("cpu", &format!(r"^llama-[^-]+-bin-win-cpu-{arch}\.zip$")),
        "cpu" => one("cpu", &format!(r"^llama-[^-]+-bin-{os}-{arch}\.tar\.gz$")),
        other => bail!("unknown backend {other} (use cuda, vulkan, metal or cpu)"),
    };
    picked.with_context(|| {
        format!("no {want} build of llama.cpp for {os}-{arch}; try --backend or --llama-dir")
    })
}

fn pick_cuda(names: &[&str], os: &str, arch: &str, driver: (u32, u32)) -> Option<Picked> {
    let re = Regex::new(&format!(
        r"^llama-[^-]+-bin-{os}-cuda-(\d+)\.(\d+)-{arch}\.(zip|tar\.gz)$"
    ))
    .unwrap();
    let (name, ver) = names
        .iter()
        .filter_map(|n| {
            let c = re.captures(n)?;
            Some((
                n.to_string(),
                (c[1].parse::<u32>().ok()?, c[2].parse::<u32>().ok()?),
            ))
        })
        .filter(|(_, v)| *v <= driver)
        .max_by_key(|(_, v)| *v)?;
    let mut files = vec![name];
    // Windows builds need the CUDA runtime DLLs, shipped as a separate archive
    let rt = format!("-bin-{os}-cuda-{}.{}-{arch}.", ver.0, ver.1);
    if let Some(r) = names
        .iter()
        .find(|n| n.starts_with("cudart-") && n.contains(&rt))
    {
        files.push(r.to_string());
    }
    Some(Picked {
        backend: format!("cuda-{}.{}", ver.0, ver.1),
        files,
    })
}

#[derive(Debug, Clone, Default)]
pub struct Speed {
    pub prompt: f64,
    pub generate: f64,
}

pub fn bench(l: &Llama, model: &Path, reps: u32) -> Result<Speed> {
    let out = Command::new(l.bin("llama-bench"))
        .arg("-m")
        .arg(model)
        .args([
            "-o",
            "jsonl",
            "-p",
            "512",
            "-n",
            "128",
            "-ngl",
            "999",
            "-r",
            &reps.to_string(),
        ])
        .output()
        .context("running llama-bench")?;
    if !out.status.success() {
        bail!("llama-bench failed:\n{}", tail(&out.stderr));
    }
    parse_bench(&String::from_utf8_lossy(&out.stdout))
}

pub fn parse_bench(jsonl: &str) -> Result<Speed> {
    let mut s = Speed::default();
    for line in jsonl.lines().filter(|l| l.trim_start().starts_with('{')) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let ts = v["avg_ts"].as_f64().unwrap_or(0.0);
        if v["n_gen"].as_u64().unwrap_or(0) > 0 {
            s.generate = ts;
        } else if v["n_prompt"].as_u64().unwrap_or(0) > 0 {
            s.prompt = ts;
        }
    }
    if s.generate == 0.0 {
        bail!("llama-bench gave no generation result");
    }
    Ok(s)
}

#[derive(Debug, Clone, Default)]
pub struct Quality {
    pub mean_kld: f64,
    pub p99_kld: f64,
    pub same_top: f64,
    pub ppl_ratio: f64,
}

pub struct Eval<'a> {
    pub text: &'a Path,
    pub ctx: u32,
    pub chunks: u32,
}

/// Saves the reference model's logits so every quant can be compared against them.
pub fn save_reference(l: &Llama, model: &Path, ev: &Eval, logits: &Path) -> Result<()> {
    let out = perplexity(l, model, ev)
        .arg("--kl-divergence-base")
        .arg(logits)
        .output()?;
    if !out.status.success() {
        bail!(
            "llama-perplexity failed on the reference:\n{}",
            tail(&out.stderr)
        );
    }
    Ok(())
}

pub fn compare(l: &Llama, model: &Path, ev: &Eval, logits: &Path) -> Result<Quality> {
    let out = perplexity(l, model, ev)
        .arg("--kl-divergence-base")
        .arg(logits)
        .arg("--kl-divergence")
        .output()?;
    let text =
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        bail!("llama-perplexity failed:\n{}", tail(text.as_bytes()));
    }
    parse_kld(&text)
}

fn perplexity(l: &Llama, model: &Path, ev: &Eval) -> Command {
    let mut c = Command::new(l.bin("llama-perplexity"));
    c.arg("-m").arg(model).arg("-f").arg(ev.text);
    c.args([
        "-c",
        &ev.ctx.to_string(),
        "--chunks",
        &ev.chunks.to_string(),
        "-ngl",
        "999",
    ]);
    c
}

static MEAN_KLD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Mean\s+KLD:\s+([\d.]+)").unwrap());
static P99_KLD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"99\.0%\s+KLD:\s+([\d.]+)").unwrap());
static SAME_TOP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"Same top p:\s+([\d.]+)").unwrap());
static PPL_RATIO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s+([\d.]+)").unwrap());

pub fn parse_kld(text: &str) -> Result<Quality> {
    let get = |re: &Regex, what: &str| -> Result<f64> {
        let c = re
            .captures(text)
            .with_context(|| format!("no {what} in llama-perplexity output"))?;
        Ok(c[1].parse()?)
    };
    Ok(Quality {
        mean_kld: get(&MEAN_KLD, "mean KLD")?,
        p99_kld: get(&P99_KLD, "99% KLD")?,
        same_top: get(&SAME_TOP, "same-top-token rate")?,
        ppl_ratio: get(&PPL_RATIO, "perplexity ratio")?,
    })
}

fn tail(bytes: &[u8]) -> String {
    let s = String::from_utf8_lossy(bytes);
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(15)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASSETS: &[&str] = &[
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.4-x64.zip",
        "llama-b11829-bin-macos-arm64.tar.gz",
        "llama-b11829-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "llama-b11829-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "llama-b11829-bin-ubuntu-vulkan-x64.tar.gz",
        "llama-b11829-bin-ubuntu-x64.tar.gz",
        "llama-b11829-bin-win-cpu-x64.zip",
        "llama-b11829-bin-win-cuda-12.4-x64.zip",
        "llama-b11829-bin-win-cuda-13.4-x64.zip",
        "llama-b11829-bin-win-vulkan-x64.zip",
    ];

    #[test]
    fn picks_newest_cuda_the_driver_supports() {
        let p = pick_assets(ASSETS, "windows", "x86_64", None, Some((13, 4))).unwrap();
        assert_eq!(
            p.files,
            [
                "llama-b11829-bin-win-cuda-13.4-x64.zip",
                "cudart-llama-bin-win-cuda-13.4-x64.zip"
            ]
        );
        let p = pick_assets(ASSETS, "windows", "x86_64", None, Some((12, 9))).unwrap();
        assert_eq!(p.backend, "cuda-12.4");
        let p = pick_assets(ASSETS, "linux", "x86_64", None, Some((13, 0))).unwrap();
        assert_eq!(p.files, ["llama-b11829-bin-ubuntu-cuda-12.8-x64.tar.gz"]);
    }

    #[test]
    fn falls_back_without_cuda() {
        assert_eq!(
            pick_assets(ASSETS, "linux", "x86_64", None, None)
                .unwrap()
                .backend,
            "vulkan"
        );
        assert_eq!(
            pick_assets(ASSETS, "macos", "aarch64", None, None)
                .unwrap()
                .backend,
            "metal"
        );
        let p = pick_assets(ASSETS, "windows", "x86_64", Some("cpu"), None).unwrap();
        assert_eq!(p.files, ["llama-b11829-bin-win-cpu-x64.zip"]);
        assert!(pick_assets(ASSETS, "windows", "x86_64", None, Some((11, 0))).is_err());
    }

    #[test]
    fn parses_bench_jsonl() {
        let out = r#"{"n_prompt": 512, "n_gen": 0, "avg_ts": 48808.99}
{"n_prompt": 0, "n_gen": 128, "avg_ts": 708.63}"#;
        let s = parse_bench(out).unwrap();
        assert_eq!((s.prompt, s.generate), (48808.99, 708.63));
    }

    #[test]
    fn parses_kld_stats() {
        let out = "====== Perplexity statistics ======
Mean PPL(Q)                   :  24.855655 ±   1.683320
Mean PPL(base)                :  23.469444 ±   1.551087
Mean PPL(Q)/PPL(base)         :   1.059064 ±   0.010845

====== KL divergence statistics ======
Mean    KLD:   0.068632 ±   0.001942
Maximum KLD:   2.521479
99.9%   KLD:   0.875791
99.0%   KLD:   0.421615
Median  KLD:   0.045818
Same top p: 84.431 ± 0.718 %";
        let q = parse_kld(out).unwrap();
        assert_eq!(
            (q.mean_kld, q.p99_kld, q.same_top, q.ppl_ratio),
            (0.068632, 0.421615, 84.431, 1.059064)
        );
    }
}
