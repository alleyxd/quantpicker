//! Just enough of the GGUF format to read a model's shape from the start of the file, so we can work out how much
//! memory its KV cache needs without downloading the weights.

use anyhow::{Result, bail};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Ints(Vec<i64>),
    Other,
}

/// Why parsing stopped early: the metadata goes on past the bytes we have.
#[derive(Debug)]
pub struct NeedMore;

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], NeedMore> {
        let end = self.pos.checked_add(n).ok_or(NeedMore)?;
        if end > self.b.len() {
            return Err(NeedMore);
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, NeedMore> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, NeedMore> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, NeedMore> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn scalar(&mut self, ty: u32) -> Result<Value, NeedMore> {
        Ok(match ty {
            0 => Value::Int(self.take(1)?[0] as i64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::Int(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::Int(self.u32()? as i64),
            5 => Value::Int(self.u32()? as i32 as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            10 => Value::Int(self.u64()? as i64),
            11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            _ => return Err(NeedMore),
        })
    }
    fn value(&mut self, ty: u32) -> Result<Value, NeedMore> {
        if ty != 9 {
            return self.scalar(ty);
        }
        let inner = self.u32()?;
        let n = self.u64()? as usize;
        let mut ints = Vec::new();
        for _ in 0..n {
            if let Value::Int(i) = self.scalar(inner)? {
                ints.push(i);
            }
        }
        // per-layer head counts are arrays of ints; anything else (token lists) we only skip over
        Ok(if inner <= 5 || inner == 10 || inner == 11 {
            Value::Ints(ints)
        } else {
            Value::Other
        })
    }
}

/// Reads metadata key/value pairs until `want` returns true for the keys seen so far, or the data runs out.
pub fn metadata(
    bytes: &[u8],
    mut want: impl FnMut(&[(String, Value)]) -> bool,
) -> Result<Vec<(String, Value)>> {
    let mut r = Reader { b: bytes, pos: 0 };
    let short = |_| anyhow::anyhow!("file too short");
    if r.take(4).map_err(short)? != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = r.u32().map_err(short)?;
    if version < 2 {
        bail!("GGUF version {version} is too old");
    }
    let _tensors = r.u64().map_err(short)?;
    let count = r.u64().map_err(short)?;
    let mut kv = Vec::new();
    for _ in 0..count {
        let Ok(key) = r.string() else { break };
        let Ok(ty) = r.u32() else { break };
        let Ok(value) = r.value(ty) else { break };
        kv.push((key, value));
        if want(&kv) {
            break;
        }
    }
    Ok(kv)
}

/// What the KV cache needs per token, from a model's GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub arch: String,
    pub layers: u64,
    /// K plus V elements per token, summed over layers
    pub kv_per_token: u64,
}

fn int(kv: &[(String, Value)], key: &str) -> Option<i64> {
    kv.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
}

fn per_layer(kv: &[(String, Value)], key: &str, layers: usize) -> Option<Vec<i64>> {
    kv.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| match v {
            Value::Int(i) => Some(vec![*i; layers]),
            Value::Ints(xs) if xs.len() == layers => Some(xs.clone()),
            _ => None,
        })
}

pub fn shape(kv: &[(String, Value)]) -> Option<Shape> {
    let arch = kv
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| match v {
            Value::Str(s) => Some(s.clone()),
            _ => None,
        })?;
    let key = |k: &str| format!("{arch}.{k}");
    // models with compressed (MLA) attention cache something else entirely; don't guess
    if int(kv, &key("attention.kv_lora_rank")).is_some() {
        return None;
    }
    let layers = int(kv, &key("block_count"))? as usize;
    let heads = per_layer(kv, &key("attention.head_count"), layers)?;
    let kv_heads =
        per_layer(kv, &key("attention.head_count_kv"), layers).unwrap_or_else(|| heads.clone());
    let embd = int(kv, &key("embedding_length"))?;
    let k_len = int(kv, &key("attention.key_length"));
    let v_len = int(kv, &key("attention.value_length"));
    let mut total = 0i64;
    for (h, kvh) in heads.iter().zip(&kv_heads) {
        if *h == 0 || *kvh == 0 {
            continue; // layers without attention (some hybrid models) have no KV cache
        }
        let head_dim = embd / h;
        total += kvh * (k_len.unwrap_or(head_dim) + v_len.unwrap_or(head_dim));
    }
    Some(Shape {
        arch,
        layers: layers as u64,
        kv_per_token: total as u64,
    })
}

/// True once the whole architecture section has been read. Converters write the model's own keys before the
/// tokenizer, and optional ones like `key_length` can come after the basic shape, so we wait for the tokenizer.
pub fn has_shape_keys(kv: &[(String, Value)]) -> bool {
    kv.iter().any(|(k, _)| k == "general.architecture")
        && kv.iter().any(|(k, _)| k.starts_with("tokenizer."))
}

/// KV cache bytes for `context` tokens with 16-bit cache entries, llama.cpp's default.
pub fn kv_bytes(shape: &Shape, context: u64) -> u64 {
    shape.kv_per_token * context * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(b: &mut Vec<u8>, x: &str) {
        b.extend((x.len() as u64).to_le_bytes());
        b.extend(x.as_bytes());
    }

    fn file(entries: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend((entries.len() as u64).to_le_bytes());
        for (k, ty, v) in entries {
            s(&mut b, k);
            b.extend(ty.to_le_bytes());
            b.extend(v);
        }
        b
    }

    fn string(x: &str) -> Vec<u8> {
        let mut b = Vec::new();
        s(&mut b, x);
        b
    }

    fn qwen3_8b() -> Vec<u8> {
        file(&[
            ("general.architecture", 8, string("qwen3")),
            ("qwen3.block_count", 4, 36u32.to_le_bytes().to_vec()),
            ("qwen3.embedding_length", 4, 4096u32.to_le_bytes().to_vec()),
            (
                "qwen3.attention.head_count",
                4,
                32u32.to_le_bytes().to_vec(),
            ),
            (
                "qwen3.attention.head_count_kv",
                4,
                8u32.to_le_bytes().to_vec(),
            ),
            (
                "qwen3.attention.key_length",
                4,
                128u32.to_le_bytes().to_vec(),
            ),
            (
                "qwen3.attention.value_length",
                4,
                128u32.to_le_bytes().to_vec(),
            ),
        ])
    }

    #[test]
    fn reads_qwen3_shape() {
        let kv = metadata(&qwen3_8b(), |_| false).unwrap();
        let sh = shape(&kv).unwrap();
        assert_eq!(sh.layers, 36);
        assert_eq!(sh.kv_per_token, 36 * 8 * 256);
        // 36 layers x 8 KV heads x 128 dims x (K+V) x 2 bytes x 32768 tokens = 4.5 GiB
        assert_eq!(kv_bytes(&sh, 32768), 4_831_838_208);
    }

    #[test]
    fn explicit_key_length_beats_embedding_over_heads() {
        // Qwen3-0.6B: 1024 / 16 heads would be 64, but its heads are 128 wide; llama.cpp reports 3584 MiB at 32K
        let full = file(&[
            ("general.architecture", 8, string("qwen3")),
            ("qwen3.block_count", 4, 28u32.to_le_bytes().to_vec()),
            ("qwen3.embedding_length", 4, 1024u32.to_le_bytes().to_vec()),
            (
                "qwen3.attention.head_count",
                4,
                16u32.to_le_bytes().to_vec(),
            ),
            (
                "qwen3.attention.head_count_kv",
                4,
                8u32.to_le_bytes().to_vec(),
            ),
            (
                "qwen3.attention.key_length",
                4,
                128u32.to_le_bytes().to_vec(),
            ),
            (
                "qwen3.attention.value_length",
                4,
                128u32.to_le_bytes().to_vec(),
            ),
            ("tokenizer.ggml.model", 8, string("gpt2")),
        ]);
        let kv = metadata(&full, has_shape_keys).unwrap();
        assert_eq!(kv_bytes(&shape(&kv).unwrap(), 32768), 3584 << 20);
    }

    #[test]
    fn stops_cleanly_when_bytes_run_out() {
        let full = qwen3_8b();
        let kv = metadata(&full[..full.len() - 3], |_| false).unwrap();
        assert_eq!(kv.len(), 6); // the last entry was cut off, the rest is intact
        assert!(metadata(b"NOPE", |_| false).is_err());
    }

    #[test]
    fn per_layer_kv_heads_and_skipped_layers() {
        let mut heads = 5u32.to_le_bytes().to_vec(); // element type: int32
        heads.extend(3u64.to_le_bytes());
        for h in [4i32, 0, 2] {
            heads.extend(h.to_le_bytes());
        }
        let kv = metadata(
            &file(&[
                ("general.architecture", 8, string("x")),
                ("x.block_count", 4, 3u32.to_le_bytes().to_vec()),
                ("x.embedding_length", 4, 512u32.to_le_bytes().to_vec()),
                ("x.attention.head_count", 4, 8u32.to_le_bytes().to_vec()),
                ("x.attention.head_count_kv", 9, heads),
            ]),
            |_| false,
        )
        .unwrap();
        // head_dim 64; layers with 4, 0 and 2 KV heads -> (4 + 2) x 64 x 2
        assert_eq!(shape(&kv).unwrap().kv_per_token, 6 * 64 * 2);
    }

    #[test]
    fn mla_models_are_not_guessed() {
        let kv = metadata(
            &file(&[
                ("general.architecture", 8, string("deepseek2")),
                (
                    "deepseek2.attention.kv_lora_rank",
                    4,
                    512u32.to_le_bytes().to_vec(),
                ),
            ]),
            |_| false,
        )
        .unwrap();
        assert!(shape(&kv).is_none());
    }
}
