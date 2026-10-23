use comfy_table::{Cell, Table, presets::UTF8_FULL_CONDENSED};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Row {
    pub quant: String,
    pub size: u64,
    pub fits: bool,
    pub reference: bool,
    pub gen_tps: Option<f64>,
    pub prompt_tps: Option<f64>,
    pub mean_kld: Option<f64>,
    pub p99_kld: Option<f64>,
    pub same_top: Option<f64>,
    pub ppl_ratio: Option<f64>,
    pub error: Option<String>,
}

/// Rough reading of mean KL divergence against the reference. The cutoffs are a guide, not a law.
pub fn verdict(kld: f64) -> &'static str {
    match kld {
        k if k < 0.01 => "near lossless",
        k if k < 0.03 => "very good",
        k if k < 0.1 => "good",
        k if k < 0.3 => "noticeable loss",
        _ => "heavy loss",
    }
}

/// The fastest quant that fits and stays under `max_kld`. If none qualify, the most accurate one that fits.
pub fn recommend(rows: &[Row], max_kld: f64) -> Option<&Row> {
    let measured: Vec<&Row> = rows
        .iter()
        .filter(|r| r.fits && r.mean_kld.is_some() && r.gen_tps.is_some())
        .collect();
    let good = measured
        .iter()
        .filter(|r| r.mean_kld.unwrap() <= max_kld)
        .max_by(|a, b| {
            a.gen_tps
                .partial_cmp(&b.gen_tps)
                .unwrap()
                .then(b.size.cmp(&a.size))
        });
    good.or_else(|| {
        measured
            .iter()
            .min_by(|a, b| a.mean_kld.partial_cmp(&b.mean_kld).unwrap())
    })
    .copied()
}

pub const GIB: f64 = (1u64 << 30) as f64;

pub fn gb(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB)
}

/// 1234567 -> "1.2M"
pub fn count(n: u64) -> String {
    match n {
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.0}K", n as f64 / 1e3),
        n => n.to_string(),
    }
}

pub fn repo_table(repos: &[crate::hf::Repo]) -> Table {
    let mut t = Table::new();
    t.load_style(UTF8_FULL_CONDENSED);
    t.set_header(["repo", "downloads", "likes"]);
    for r in repos {
        t.add_row([r.id.clone(), count(r.downloads), count(r.likes)]);
    }
    t
}

pub fn quant_table(quants: &[crate::hf::Quant], usable: u64, tested: &[&str]) -> Table {
    let mut t = Table::new();
    t.load_style(UTF8_FULL_CONDENSED);
    t.set_header(["quant", "size", "fits", "measured by default"]);
    for q in quants {
        t.add_row([
            q.name.clone(),
            gb(q.size),
            if q.size <= usable { "yes" } else { "no" }.to_string(),
            if q.name == "Q8_0" {
                "reference"
            } else if tested.contains(&q.name.as_str()) {
                "tested"
            } else {
                ""
            }
            .to_string(),
        ]);
    }
    t
}

pub fn table(rows: &[Row], best: Option<&Row>) -> Table {
    let mut t = Table::new();
    t.load_style(UTF8_FULL_CONDENSED);
    t.set_header([
        "quant",
        "size",
        "gen tok/s",
        "prompt tok/s",
        "mean KLD",
        "99% KLD",
        "same top",
        "PPL",
        "verdict",
    ]);
    for r in rows {
        let name = if best.is_some_and(|b| b.quant == r.quant) {
            format!("★ {}", r.quant)
        } else {
            r.quant.clone()
        };
        let f =
            |v: Option<f64>, p: usize| v.map(|x| format!("{x:.p$}")).unwrap_or_else(|| "-".into());
        let verdict = if r.reference {
            "reference".to_string()
        } else if !r.fits {
            "doesn't fit".to_string()
        } else if let Some(e) = &r.error {
            format!("failed: {}", e.lines().next().unwrap_or(""))
        } else {
            r.mean_kld.map(verdict).unwrap_or("-").to_string()
        };
        t.add_row(vec![
            Cell::new(name),
            Cell::new(gb(r.size)),
            Cell::new(f(r.gen_tps, 0)),
            Cell::new(f(r.prompt_tps, 0)),
            Cell::new(if r.reference {
                "0".into()
            } else {
                f(r.mean_kld, 4)
            }),
            Cell::new(if r.reference {
                "0".into()
            } else {
                f(r.p99_kld, 3)
            }),
            Cell::new(if r.reference {
                "100%".into()
            } else {
                r.same_top.map(|v| format!("{v:.1}%")).unwrap_or("-".into())
            }),
            Cell::new(if r.reference {
                "-".into()
            } else {
                r.ppl_ratio
                    .map(|v| format!("{:+.1}%", (v - 1.0) * 100.0))
                    .unwrap_or("-".into())
            }),
            Cell::new(verdict),
        ]);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(q: &str, size: u64, fits: bool, tps: f64, kld: f64) -> Row {
        Row {
            quant: q.into(),
            size,
            fits,
            reference: false,
            gen_tps: Some(tps),
            prompt_tps: None,
            mean_kld: Some(kld),
            p99_kld: None,
            same_top: None,
            ppl_ratio: None,
            error: None,
        }
    }

    #[test]
    fn picks_fastest_under_the_limit() {
        let rows = [
            row("Q3_K_M", 3, true, 160.0, 0.12),
            row("Q4_K_M", 4, true, 150.0, 0.03),
            row("Q6_K", 6, true, 120.0, 0.005),
        ];
        assert_eq!(recommend(&rows, 0.05).unwrap().quant, "Q4_K_M");
        assert_eq!(recommend(&rows, 0.2).unwrap().quant, "Q3_K_M");
    }

    #[test]
    fn ignores_quants_that_dont_fit() {
        let rows = [
            row("Q4_K_M", 4, true, 150.0, 0.08),
            row("Q6_K", 6, false, 0.0, 0.005),
        ];
        // nothing under the limit fits, so take the most accurate one that does
        assert_eq!(recommend(&rows, 0.05).unwrap().quant, "Q4_K_M");
    }

    #[test]
    fn counts() {
        assert_eq!(count(1_234_567), "1.2M");
        assert_eq!(count(64_200), "64K");
        assert_eq!(count(182), "182");
    }

    #[test]
    fn verdicts() {
        assert_eq!(verdict(0.004), "near lossless");
        assert_eq!(verdict(0.07), "good");
        assert_eq!(verdict(0.5), "heavy loss");
    }
}
