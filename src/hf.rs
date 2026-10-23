use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use serde::Deserialize;

/// Hugging Face base URL. `HF_ENDPOINT` points it at a mirror, the same variable huggingface_hub reads.
fn hf() -> String {
    endpoint(std::env::var("HF_ENDPOINT").ok())
}

fn endpoint(var: Option<String>) -> String {
    match var {
        Some(v) if !v.trim().is_empty() => v.trim().trim_end_matches('/').to_string(),
        _ => "https://huggingface.co".to_string(),
    }
}

#[derive(Deserialize)]
struct Entry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
}

/// One quantization of a model. Large quants are often split into several files.
#[derive(Debug, Clone)]
pub struct Quant {
    pub name: String,
    pub files: Vec<String>,
    pub size: u64,
}

static SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"-\d{5}-of-\d{5}$").unwrap());
static QUANT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))$")
        .unwrap()
});

/// Pulls the quant label out of a GGUF file name, e.g. "Qwen3-8B-Q4_K_M-00001-of-00002.gguf" -> "Q4_K_M".
pub fn quant_name(path: &str) -> Option<String> {
    let file = path.rsplit('/').next()?;
    let stem = file
        .strip_suffix(".gguf")
        .or_else(|| file.strip_suffix(".GGUF"))?;
    let stem = SPLIT.replace(stem, "");
    QUANT.captures(&stem).map(|c| c[1].to_uppercase())
}

fn client(token: Option<&str>) -> Result<reqwest::blocking::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(t) = token {
        headers.insert("Authorization", format!("Bearer {t}").parse()?);
    }
    Ok(reqwest::blocking::Client::builder()
        .user_agent(concat!("quantpicker/", env!("CARGO_PKG_VERSION")))
        .default_headers(headers)
        .timeout(None)
        .build()?)
}

pub fn list_quants(repo: &str, token: Option<&str>) -> Result<Vec<Quant>> {
    let url = format!("{}/api/models/{repo}/tree/main?recursive=true", hf());
    let resp = client(token)?.get(&url).send()?;
    if resp.status() == 404 || resp.status() == 401 {
        bail!("can't see {repo} on Hugging Face (check the name, or set HF_TOKEN for gated repos)");
    }
    let entries: Vec<Entry> = resp.error_for_status()?.json()?;
    Ok(group(entries))
}

#[derive(Debug, Deserialize)]
pub struct Repo {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
}

/// GGUF repos on Hugging Face matching `query`. Repos named after the query come first, then by downloads.
pub fn search(query: &str, limit: usize, token: Option<&str>) -> Result<Vec<Repo>> {
    let resp = client(token)?
        .get(format!("{}/api/models", hf()))
        .query(&[
            ("search", query),
            ("filter", "gguf"),
            ("sort", "downloads"),
            ("direction", "-1"),
            ("limit", &(limit * 3).to_string()),
        ])
        .send()?;
    let mut repos: Vec<Repo> = resp.error_for_status()?.json()?;
    rank(&mut repos, query);
    repos.truncate(limit);
    Ok(repos)
}

/// Hugging Face sorts purely by downloads, so "qwen3 8b" puts Qwen3-VL-8B above Qwen3-8B. Prefer repos whose
/// name starts with the query, then ones that contain it, keeping download order within each group.
fn rank(repos: &mut [Repo], query: &str) {
    let q = query
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
        .to_lowercase();
    repos.sort_by_key(|r| {
        let name = r.id.rsplit('/').next().unwrap_or(&r.id).to_lowercase();
        let group = if name.starts_with(&q) {
            0
        } else if name.contains(&q) {
            1
        } else {
            2
        };
        (group, std::cmp::Reverse(r.downloads))
    });
}

/// What to search for when someone passes a repo without GGUF files, e.g. "Qwen/Qwen3-8B" -> "Qwen3-8B".
pub fn search_term(repo: &str) -> &str {
    repo.rsplit('/').next().unwrap_or(repo)
}

fn group(entries: Vec<Entry>) -> Vec<Quant> {
    let mut by_name: BTreeMap<String, Quant> = BTreeMap::new();
    for e in entries.into_iter().filter(|e| e.kind == "file") {
        // mmproj files are vision projectors, not the model itself
        if e.path.to_lowercase().contains("mmproj") {
            continue;
        }
        let Some(name) = quant_name(&e.path) else {
            continue;
        };
        let q = by_name.entry(name.clone()).or_insert(Quant {
            name,
            files: vec![],
            size: 0,
        });
        q.size += e.size;
        q.files.push(e.path);
    }
    let mut quants: Vec<Quant> = by_name.into_values().collect();
    for q in &mut quants {
        q.files.sort();
    }
    quants.sort_by_key(|q| q.size);
    quants
}

/// Downloads every file of a quant into `dir`, resuming partial downloads. Returns the path of the first
/// file, which is what llama.cpp wants for split models.
pub fn download(repo: &str, quant: &Quant, dir: &Path, token: Option<&str>) -> Result<PathBuf> {
    let client = client(token)?;
    let mut first = None;
    for file in &quant.files {
        let dest = dir.join(file.rsplit('/').next().unwrap());
        first.get_or_insert(dest.clone());
        if dest.exists() {
            continue;
        }
        let part = dest.with_extension("gguf.part");
        let have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let url = format!("{}/{repo}/resolve/main/{file}", hf());
        let mut req = client.get(&url);
        if have > 0 {
            req = req.header("Range", format!("bytes={have}-"));
        }
        let mut resp = req
            .send()?
            .error_for_status()
            .with_context(|| format!("downloading {file}"))?;
        let resumed = resp.status() == 206;
        let total = resp.content_length().unwrap_or(0) + if resumed { have } else { 0 };

        let bar = ProgressBar::new(total);
        bar.set_style(ProgressStyle::with_template(
            "  {msg} {bar:30} {bytes}/{total_bytes} {bytes_per_sec}",
        )?);
        bar.set_message(file.rsplit('/').next().unwrap().to_string());
        bar.set_position(if resumed { have } else { 0 });

        let mut out = OpenOptions::new()
            .create(true)
            .write(true)
            .append(resumed)
            .truncate(!resumed)
            .open(&part)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = resp.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            bar.inc(n as u64);
        }
        bar.finish_and_clear();
        fs::rename(&part, &dest)?;
    }
    first.context("quant has no files")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_defaults_and_mirrors() {
        assert_eq!(endpoint(None), "https://huggingface.co");
        assert_eq!(endpoint(Some("  ".into())), "https://huggingface.co");
        assert_eq!(
            endpoint(Some("https://hf-mirror.com/".into())),
            "https://hf-mirror.com"
        );
    }

    #[test]
    fn parses_quant_names() {
        assert_eq!(
            quant_name("Qwen3-8B-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            quant_name("Qwen3-0.6B-UD-Q4_K_XL.gguf").as_deref(),
            Some("UD-Q4_K_XL")
        );
        assert_eq!(quant_name("model.IQ4_XS.gguf").as_deref(), Some("IQ4_XS"));
        assert_eq!(quant_name("Qwen3-0.6B-BF16.gguf").as_deref(), Some("BF16"));
        assert_eq!(
            quant_name("big/Llama-70B-Q8_0-00002-of-00003.gguf").as_deref(),
            Some("Q8_0")
        );
        assert_eq!(
            quant_name("gpt-oss-20b-MXFP4.gguf").as_deref(),
            Some("MXFP4")
        );
        assert_eq!(quant_name("README.md"), None);
        assert_eq!(quant_name("tokenizer.gguf"), None);
    }

    #[test]
    fn ranks_name_matches_first() {
        let r = |id: &str, downloads| Repo {
            id: id.into(),
            downloads,
            likes: 0,
        };
        let mut repos = vec![
            r("Qwen/Qwen3-VL-8B-Instruct-GGUF", 900),
            r("someone/Parable-Qwen3-8B-GGUF", 200),
            r("Qwen/Qwen3-8B-GGUF", 500),
            r("unsloth/Qwen3-8B-GGUF", 600),
        ];
        rank(&mut repos, "qwen3 8b");
        let ids: Vec<&str> = repos.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "unsloth/Qwen3-8B-GGUF",
                "Qwen/Qwen3-8B-GGUF",
                "someone/Parable-Qwen3-8B-GGUF",
                "Qwen/Qwen3-VL-8B-Instruct-GGUF",
            ]
        );
    }

    #[test]
    fn groups_split_files() {
        let e = |p: &str, s| Entry {
            kind: "file".into(),
            path: p.into(),
            size: s,
        };
        let quants = group(vec![
            e("Q8_0/m-Q8_0-00002-of-00002.gguf", 40),
            e("Q8_0/m-Q8_0-00001-of-00002.gguf", 50),
            e("m-Q4_K_M.gguf", 30),
            e("mmproj-m-F16.gguf", 5),
            e("README.md", 1),
        ]);
        assert_eq!(quants.len(), 2);
        assert_eq!(quants[0].name, "Q4_K_M");
        assert_eq!(quants[1].size, 90);
        assert_eq!(quants[1].files[0], "Q8_0/m-Q8_0-00001-of-00002.gguf");
    }
}
