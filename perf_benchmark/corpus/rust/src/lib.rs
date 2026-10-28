//! Offline hash prompt inflation based on blitz-serving/trace-replayer.
//!
//! Upstream commit 84c9193ac67ebbf0e08db161bc9b048d597c2f80,
//! src/token_sampler.rs and src/dataset.rs, Apache-2.0. The v2 batch sampler,
//! bounded full-block channel and Bailian/Mooncake hash-cache rules are retained.
//! Partial blocks are requested lazily so a 512-token trace does not prewarm
//! 511 separate queues. Python owns source parsing and atomic Parquet publishing.

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, SendTimeoutError, Sender};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::Rng;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tokenizers::Tokenizer;

const PAD: u32 = 151643;
const BOS: u32 = 151644;
const EOS: u32 = 151645;
const WAIT: Duration = Duration::from_millis(20);

fn validate_tokenizer(config_path: &str) -> Result<(), String> {
    let raw = fs::read_to_string(config_path).map_err(|e| format!("tokenizer config: {e}"))?;
    let config: Value = serde_json::from_str(&raw).map_err(|e| format!("tokenizer config: {e}"))?;
    if config["tokenizer_class"] != "Qwen2Tokenizer" {
        return Err("text block sampling requires Qwen2Tokenizer".into());
    }
    for (id, token) in [
        (PAD, "<|endoftext|>"),
        (BOS, "<|im_start|>"),
        (EOS, "<|im_end|>"),
    ] {
        if config["added_tokens_decoder"][id.to_string()]["content"] != token {
            return Err("Qwen2 special token configuration differs from upstream replayer".into());
        }
    }
    Ok(())
}

struct BatchSampleContext {
    tokens: Vec<u32>,
    begin: usize,
    batch_size: usize,
}

impl BatchSampleContext {
    fn new(block_size: usize) -> Self {
        Self {
            tokens: Vec::new(),
            begin: 0,
            batch_size: 2048.max(block_size * 4),
        }
    }

    fn reload(&mut self, tokenizer: &Tokenizer, rng: &mut impl Rng) -> Result<(), String> {
        // Upstream BatchSampleContext.reload(): sample vocabulary IDs, then
        // decode and re-encode them so subsequent slices are valid text tokens.
        let vocab_size = tokenizer.get_vocab_size(true) as u32;
        if vocab_size == 0 {
            return Err("tokenizer vocabulary is empty".into());
        }
        let ids: Vec<u32> = (0..self.batch_size)
            .map(|_| rng.gen_range(0..vocab_size))
            .collect();
        let decoded = tokenizer.decode(&ids, false).map_err(|e| e.to_string())?;
        self.tokens = tokenizer
            .encode(decoded, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        self.begin = 0;
        Ok(())
    }
}

enum BlockOutcome {
    Exact(String),
    OtherLength(String, usize),
    EndOfContext,
}

fn sample_from_context(
    tokenizer: &Tokenizer,
    size: usize,
    ctx: &mut BatchSampleContext,
) -> Result<BlockOutcome, String> {
    if size == 1 {
        return Ok(BlockOutcome::Exact("<|endoftext|>".into()));
    }
    if size == 2 {
        return Ok(BlockOutcome::Exact("<|im_start|><|im_end|>".into()));
    }
    let begin = ctx.begin;
    let mut end = begin + size - 2;
    while end <= ctx.tokens.len() {
        let mut ids = Vec::with_capacity(end - begin + 2);
        ids.push(BOS);
        ids.extend_from_slice(&ctx.tokens[begin..end]);
        ids.push(EOS);
        let text = tokenizer.decode(&ids, false).map_err(|e| e.to_string())?;
        let count = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .len();
        if count == size {
            ctx.begin = end;
            return Ok(BlockOutcome::Exact(text));
        }
        if count > size {
            ctx.begin = end;
            return Ok(BlockOutcome::OtherLength(text, count));
        }
        end += 1;
    }
    Ok(BlockOutcome::EndOfContext)
}

fn generate_block(
    tokenizer: &Tokenizer,
    size: usize,
    ctx: &mut BatchSampleContext,
    ragged_txs: &[Sender<String>],
    rng: &mut impl Rng,
    stop: &AtomicBool,
) -> Result<String, String> {
    while !stop.load(Ordering::Acquire) {
        if ctx.begin + size > ctx.tokens.len() {
            ctx.reload(tokenizer, rng)?;
        }
        match sample_from_context(tokenizer, size, ctx)? {
            BlockOutcome::Exact(text) => return Ok(text),
            BlockOutcome::OtherLength(text, actual) => {
                // The upstream v2 sampler recycles a valid overshoot into the
                // matching partial-block channel rather than throwing it away.
                if actual > 0 && actual <= ragged_txs.len() {
                    let _ = ragged_txs[actual - 1].try_send(text);
                }
            }
            BlockOutcome::EndOfContext => ctx.reload(tokenizer, rng)?,
        }
    }
    Err("trace block producer stopped".into())
}

fn producer_loop(
    tokenizer: Tokenizer,
    block_size: usize,
    full_tx: &Sender<String>,
    ragged_commands: &Receiver<usize>,
    ragged_txs: &[Sender<String>],
    stop: Arc<AtomicBool>,
) -> Result<(), String> {
    let mut rng = rand::thread_rng();
    let mut ctx = BatchSampleContext::new(block_size);
    while !stop.load(Ordering::Acquire) {
        if let Ok(size) = ragged_commands.try_recv() {
            let block = generate_block(&tokenizer, size, &mut ctx, ragged_txs, &mut rng, &stop)?;
            let _ = ragged_txs[size - 1].try_send(block);
            continue;
        }
        let block = generate_block(
            &tokenizer, block_size, &mut ctx, ragged_txs, &mut rng, &stop,
        )?;
        let mut pending = block;
        while !stop.load(Ordering::Acquire) {
            match full_tx.send_timeout(pending, WAIT) {
                Ok(()) => break,
                Err(SendTimeoutError::Disconnected(_)) => return Ok(()),
                Err(SendTimeoutError::Timeout(block)) => pending = block,
            }
            if let Ok(size) = ragged_commands.try_recv() {
                let block =
                    generate_block(&tokenizer, size, &mut ctx, ragged_txs, &mut rng, &stop)?;
                let _ = ragged_txs[size - 1].try_send(block);
            }
        }
    }
    Ok(())
}

struct TokenSampler {
    block_size: usize,
    full_rx: Receiver<String>,
    ragged_command_tx: Sender<usize>,
    ragged_rxs: Vec<Receiver<String>>,
    stop: Arc<AtomicBool>,
    active: Arc<AtomicUsize>,
    error: Arc<Mutex<Option<String>>>,
    threads: Vec<JoinHandle<()>>,
}

impl TokenSampler {
    fn new(
        tokenizer: Tokenizer,
        block_size: usize,
        workers: usize,
        capacity: usize,
    ) -> Result<Self, String> {
        if block_size == 0 || workers == 0 || capacity == 0 {
            return Err("block_size, workers and queue_capacity must be positive".into());
        }
        let (full_tx, full_rx) = bounded(capacity);
        let (ragged_command_tx, ragged_commands) = bounded(workers * 4);
        let mut ragged_txs = Vec::with_capacity(block_size.saturating_sub(1));
        let mut ragged_rxs = Vec::with_capacity(block_size.saturating_sub(1));
        for _ in 1..block_size {
            let (tx, rx) = bounded(2);
            ragged_txs.push(tx);
            ragged_rxs.push(rx);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let active = Arc::new(AtomicUsize::new(workers));
        let error = Arc::new(Mutex::new(None));
        let mut threads = Vec::with_capacity(workers);
        for index in 0..workers {
            let (tokenizer, tx, commands, tails) = (
                tokenizer.clone(),
                full_tx.clone(),
                ragged_commands.clone(),
                ragged_txs.clone(),
            );
            let (stopped, count, failure) = (stop.clone(), active.clone(), error.clone());
            let handle = thread::Builder::new()
                .name(format!("trace-block-{index}"))
                .spawn(move || {
                    // Keep channels connected until the failure state is published.
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        producer_loop(
                            tokenizer,
                            block_size,
                            &tx,
                            &commands,
                            &tails,
                            stopped.clone(),
                        )
                    }));
                    let message = match result {
                        Ok(Ok(())) => None,
                        Ok(Err(message)) => Some(message),
                        Err(_) => Some("trace block producer panicked".into()),
                    };
                    if let Some(message) = message {
                        let mut error = failure.lock().unwrap_or_else(|e| e.into_inner());
                        if error.is_none() {
                            *error = Some(message);
                        }
                        stopped.store(true, Ordering::Release);
                    }
                    count.fetch_sub(1, Ordering::AcqRel);
                });
            match handle {
                Ok(handle) => threads.push(handle),
                Err(e) => {
                    stop.store(true, Ordering::Release);
                    for handle in threads {
                        let _ = handle.join();
                    }
                    return Err(format!("could not start trace block producer: {e}"));
                }
            }
        }
        Ok(Self {
            block_size,
            full_rx,
            ragged_command_tx,
            ragged_rxs,
            stop,
            active,
            error,
            threads,
        })
    }

    fn check(&self) -> Result<(), String> {
        let stopped = self.active.load(Ordering::Acquire) == 0 || self.stop.load(Ordering::Acquire);
        if let Some(message) = self
            .error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            return Err(message.clone());
        }
        if stopped {
            return Err("trace block producer stopped".into());
        }
        Ok(())
    }

    fn get(&self, size: usize) -> Result<String, String> {
        if size == 0 || size > self.block_size {
            return Err("invalid text block length".into());
        }
        self.check()?;
        let rx = if size == self.block_size {
            &self.full_rx
        } else {
            let rx = &self.ragged_rxs[size - 1];
            // Refill only the requested tail size, as in gen_string().
            loop {
                match self.ragged_command_tx.send_timeout(size, WAIT) {
                    Ok(()) => break,
                    Err(SendTimeoutError::Disconnected(_)) => {
                        self.check()?;
                        return Err("trace block producer stopped".into());
                    }
                    Err(SendTimeoutError::Timeout(_)) => self.check()?,
                }
            }
            rx
        };
        loop {
            match rx.recv_timeout(WAIT) {
                Ok(block) => return Ok(block),
                Err(RecvTimeoutError::Disconnected) => {
                    self.check()?;
                    return Err("trace block producer stopped".into());
                }
                Err(RecvTimeoutError::Timeout) => self.check()?,
            }
        }
    }
}

impl Drop for TokenSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

struct Inflator {
    sampler: TokenSampler,
    blocks: HashMap<u64, String>,
    partial_lengths: HashMap<u64, usize>,
    last_full_prompt: Option<(Vec<u64>, String)>,
    text_bytes: usize,
    cache_max_bytes: usize,
    hits: u64,
    misses: u64,
    partial_overwrites: u64,
    peak_cache_bytes: usize,
}

impl Inflator {
    fn cache_bytes(&self) -> usize {
        self.text_bytes
            + self.blocks.capacity() * (size_of::<(u64, String)>() + 1)
            + self.partial_lengths.capacity() * (size_of::<(u64, usize)>() + 1)
            + self.last_full_prompt.as_ref().map_or(0, |(hashes, text)| {
                hashes.capacity() * size_of::<u64>() + text.capacity()
            })
    }

    fn check_budget(&mut self) -> Result<(), String> {
        let bytes = self.cache_bytes();
        self.peak_cache_bytes = self.peak_cache_bytes.max(bytes);
        if bytes > self.cache_max_bytes {
            return Err(format!(
                "trace hash cache exceeded cache_max_bytes: {bytes} > {}",
                self.cache_max_bytes
            ));
        }
        Ok(())
    }

    fn reset_scope(&mut self) {
        self.blocks = HashMap::new();
        self.partial_lengths = HashMap::new();
        self.last_full_prompt = None;
        self.text_bytes = 0;
    }

    fn insert(&mut self, id: u64, text: String, partial: Option<usize>) -> Result<(), String> {
        self.last_full_prompt = None;
        self.text_bytes += text.capacity();
        if let Some(previous) = self.blocks.insert(id, text) {
            self.text_bytes -= previous.capacity();
        }
        if let Some(size) = partial {
            self.partial_lengths.insert(id, size);
            self.partial_overwrites += 1;
        } else {
            self.partial_lengths.remove(&id);
        }
        self.check_budget()
    }

    fn inflate(&mut self, length: usize, hashes: &[u64]) -> Result<String, String> {
        let block_size = self.sampler.block_size;
        if length == 0 || hashes.len() != length.div_ceil(block_size) {
            return Err("hash_ids count does not match input_length".into());
        }
        if length.is_multiple_of(block_size) {
            if let Some((previous_hashes, prompt)) = &self.last_full_prompt {
                if previous_hashes == hashes {
                    self.hits += hashes.len() as u64;
                    return Ok(prompt.clone());
                }
            }
        }
        let mut prompt = String::with_capacity(length.saturating_mul(8));
        let mut actual_length = 0;
        for (index, id) in hashes.iter().enumerate() {
            let size = block_size.min(length - index * block_size);
            if size < block_size {
                let block = self.sampler.get(size)?;
                prompt.push_str(&block);
                self.insert(*id, block, Some(size))?;
                self.misses += 1;
                actual_length += size;
                continue;
            }
            if let Some(block) = self.blocks.get(id) {
                if let Some(partial) = self.partial_lengths.get(id) {
                    return Err(format!(
                        "reused hash ID has a different token-block length: {partial} != {size}"
                    ));
                }
                prompt.push_str(block);
                self.hits += 1;
            } else {
                let block = self.sampler.get(size)?;
                prompt.push_str(&block);
                self.insert(*id, block, None)?;
                self.misses += 1;
            }
            actual_length += size;
        }
        if actual_length != length {
            return Err("inflated prompt token length differs from source".into());
        }
        if length.is_multiple_of(block_size) {
            self.last_full_prompt = Some((hashes.to_vec(), prompt.clone()));
            self.check_budget()?;
        }
        Ok(prompt)
    }
}

#[pyclass(module = "perf_benchmark._native")]
pub struct TraceHashInflator {
    inner: Mutex<Option<Inflator>>,
}

#[pymethods]
impl TraceHashInflator {
    #[new]
    fn new(
        tokenizer_path: &str,
        tokenizer_config_path: &str,
        block_size: usize,
        workers: usize,
        queue_capacity: usize,
        cache_max_bytes: usize,
    ) -> PyResult<Self> {
        validate_tokenizer(tokenizer_config_path).map_err(PyValueError::new_err)?;
        if cache_max_bytes == 0 {
            return Err(PyValueError::new_err("cache_max_bytes must be positive"));
        }
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| PyValueError::new_err(format!("could not load tokenizer.json: {e}")))?;
        for (id, token) in [
            (PAD, "<|endoftext|>"),
            (BOS, "<|im_start|>"),
            (EOS, "<|im_end|>"),
        ] {
            if tokenizer.token_to_id(token) != Some(id) {
                return Err(PyValueError::new_err(
                    "tokenizer.json Qwen2 special token IDs differ from upstream replayer",
                ));
            }
        }
        let sampler = TokenSampler::new(tokenizer, block_size, workers, queue_capacity)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Mutex::new(Some(Inflator {
                sampler,
                blocks: HashMap::new(),
                partial_lengths: HashMap::new(),
                last_full_prompt: None,
                text_bytes: 0,
                cache_max_bytes,
                hits: 0,
                misses: 0,
                partial_overwrites: 0,
                peak_cache_bytes: 0,
            })),
        })
    }

    fn inflate(&self, py: Python<'_>, length: usize, hash_ids: Vec<u64>) -> PyResult<String> {
        let result = py.detach(|| {
            let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            guard
                .as_mut()
                .ok_or_else(|| "trace hash inflator is closed".to_string())?
                .inflate(length, &hash_ids)
        });
        result.map_err(PyValueError::new_err)
    }

    fn reset_scope(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            guard
                .as_mut()
                .ok_or_else(|| "trace hash inflator is closed".to_string())?
                .reset_scope();
            Ok::<(), String>(())
        })
        .map_err(PyValueError::new_err)
    }

    fn stats(&self) -> PyResult<(u64, u64, u64, usize)> {
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let inner = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("trace hash inflator is closed"))?;
        Ok((
            inner.hits,
            inner.misses,
            inner.partial_overwrites,
            inner.peak_cache_bytes,
        ))
    }

    fn close(&self, py: Python<'_>) {
        py.detach(|| {
            let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            guard.take();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::BPE;

    #[test]
    fn producer_failure_is_reported_instead_of_blocking_consumer() {
        // An empty vocabulary makes random sampling impossible. The producer
        // must report its failure to a waiting consumer and remain joinable.
        for workers in [1, 2] {
            for size in [8, 16] {
                let sampler =
                    TokenSampler::new(Tokenizer::new(BPE::default()), 16, workers, 2).unwrap();
                let error = sampler.get(size).unwrap_err();
                assert!(error.contains("tokenizer vocabulary is empty"), "{error}");
                let active = sampler.active.clone();
                drop(sampler);
                assert_eq!(active.load(Ordering::Acquire), 0);
            }
        }
    }

    #[test]
    fn invalid_block_shape_is_rejected_before_sampling() {
        let sampler = TokenSampler::new(Tokenizer::new(BPE::default()), 16, 1, 2).unwrap();
        assert_eq!(sampler.get(0).unwrap_err(), "invalid text block length");
        assert_eq!(sampler.get(17).unwrap_err(), "invalid text block length");
    }
}
