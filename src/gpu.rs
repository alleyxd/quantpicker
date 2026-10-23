use std::process::Command;

#[derive(Debug, Clone)]
pub struct Gpu {
    pub name: String,
    pub total: u64,
    pub free: u64,
    /// Highest CUDA version the NVIDIA driver supports, e.g. (13, 0).
    pub cuda: Option<(u32, u32)>,
}

pub fn detect() -> Option<Gpu> {
    nvidia().or_else(apple)
}

fn nvidia() -> Option<Gpu> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    // first GPU only; llama.cpp splits across GPUs but one is the common case
    let line = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .to_string();
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    let mib = |s: &str| s.parse::<u64>().ok().map(|v| v * 1024 * 1024);
    let (total, used) = (mib(parts.get(1)?)?, mib(parts.get(2)?)?);
    let header = Command::new("nvidia-smi").output().ok()?;
    Some(Gpu {
        name: parts[0].to_string(),
        total,
        free: total.saturating_sub(used),
        cuda: cuda_version(&String::from_utf8_lossy(&header.stdout)),
    })
}

pub fn cuda_version(smi: &str) -> Option<(u32, u32)> {
    // older drivers print "CUDA Version: 12.8", newer ones "CUDA UMD Version: 13.4"
    let re = regex::Regex::new(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)").unwrap();
    let c = re.captures(smi)?;
    Some((c[1].parse().ok()?, c[2].parse().ok()?))
}

fn apple() -> Option<Gpu> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let total: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    // macOS lets the GPU use roughly three quarters of unified memory by default
    Some(Gpu {
        name: "Apple GPU (unified memory)".into(),
        total,
        free: total / 4 * 3,
        cuda: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_cuda_version() {
        let smi = "| NVIDIA-SMI 590.12   Driver Version: 590.12   CUDA Version: 13.1     |";
        assert_eq!(cuda_version(smi), Some((13, 1)));
        let smi = "| NVIDIA-SMI 616.64     KMD Version: 616.64     CUDA UMD Version: 13.4     |";
        assert_eq!(cuda_version(smi), Some((13, 4)));
        assert_eq!(cuda_version("no gpu here"), None);
    }
}
