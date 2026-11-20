//! 测量“按列稀疏读取”能否真正省下内存带宽。
//!
//! 输入通道稀疏化后，每个 token 只需读取被选中的权重列。CPU 的硬件预取器会顺着连续地址提前取数，
//! 跳过的列如果太短，仍会被预取进缓存，带宽并没有省下来。这里把一块远大于缓存的内存切成固定大小的块，
//! 每遍随机保留一部分块，多线程只读被保留的块，与全部读取的耗时比较。
//!
//! 用法：cargo run --release -p nncoss-llm --example sparse_read -- [--mb 320] [--threads 6]
//!       [--keep 0.75] [--passes 20] [--chunks 512,1024,2048,3072,8192,65536]

use std::sync::Barrier;
use std::time::Instant;

struct Options {
    megabytes: usize,
    threads: usize,
    keep: f64,
    passes: usize,
    chunks: Vec<usize>,
}

fn parse() -> Options {
    let mut options = Options {
        megabytes: 320,
        threads: 6,
        keep: 0.75,
        passes: 20,
        chunks: vec![512, 1024, 2048, 3072, 8192, 65_536],
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().unwrap_or_else(|| panic!("{flag} 缺少取值"));
        match flag.as_str() {
            "--mb" => options.megabytes = value.parse().expect("--mb"),
            "--threads" => options.threads = value.parse().expect("--threads"),
            "--keep" => options.keep = value.parse().expect("--keep"),
            "--passes" => options.passes = value.parse().expect("--passes"),
            "--chunks" => {
                options.chunks = value
                    .split(',')
                    .map(|size| size.parse().expect("--chunks"))
                    .collect()
            }
            _ => panic!("未知参数 {flag}"),
        }
    }
    options
}

/// 每遍各块是否保留；用 xorshift 生成，保证可复现且不计入计时。
fn masks(chunks: usize, passes: usize, keep: f64, seed: u64) -> Vec<Vec<bool>> {
    let mut state = seed.max(1);
    let threshold = (keep * u32::MAX as f64) as u64;
    (0..passes)
        .map(|_| {
            (0..chunks)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 32) <= threshold
                })
                .collect()
        })
        .collect()
}

fn read_chunk(chunk: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { read_chunk_avx2(chunk) };
    }
    read_chunk_scalar(chunk)
}

/// 每 64 字节一次两条 256 位读取，累加到四个寄存器，保证瓶颈在内存而不在计算。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn read_chunk_avx2(chunk: &[u8]) -> u64 {
    use std::arch::x86_64::*;
    let mut lanes = [_mm256_setzero_si256(); 4];
    let blocks = chunk.len() / 128;
    let pointer = chunk.as_ptr().cast::<__m256i>();
    for block in 0..blocks {
        for (lane, total) in lanes.iter_mut().enumerate() {
            let value = unsafe { _mm256_loadu_si256(pointer.add(block * 4 + lane)) };
            *total = _mm256_add_epi64(*total, value);
        }
    }
    let sum = _mm256_add_epi64(
        _mm256_add_epi64(lanes[0], lanes[1]),
        _mm256_add_epi64(lanes[2], lanes[3]),
    );
    let mut stored = [0_u64; 4];
    unsafe { _mm256_storeu_si256(stored.as_mut_ptr().cast(), sum) };
    stored
        .iter()
        .fold(0_u64, |total, lane| total.wrapping_add(*lane))
        .wrapping_add(read_chunk_scalar(&chunk[blocks * 128..]))
}

fn read_chunk_scalar(chunk: &[u8]) -> u64 {
    let mut lanes = [0_u64; 4];
    for block in chunk.chunks_exact(32) {
        for (lane, bytes) in lanes.iter_mut().zip(block.chunks_exact(8)) {
            *lane = lane.wrapping_add(u64::from_le_bytes(bytes.try_into().expect("8 bytes")));
        }
    }
    lanes
        .iter()
        .fold(0, |total, lane| total.wrapping_add(*lane))
}

#[cfg(target_arch = "x86_64")]
fn prefetch(pointer: *const u8) {
    use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
    unsafe { _mm_prefetch::<_MM_HINT_T0>(pointer.cast()) };
}

#[cfg(not(target_arch = "x86_64"))]
fn prefetch(_pointer: *const u8) {}

/// 返回所有遍的总耗时（秒）。software_prefetch 时，读当前块前先预取下一个保留块的开头两行。
fn run(
    buffer: &[u8],
    size: usize,
    threads: usize,
    masks: &[Vec<bool>],
    software_prefetch: bool,
) -> f64 {
    let chunks = buffer.len() / size;
    let barrier = Barrier::new(threads + 1);
    let mut elapsed = 0.0;
    std::thread::scope(|scope| {
        for thread in 0..threads {
            let barrier = &barrier;
            scope.spawn(move || {
                let range = thread * chunks / threads..(thread + 1) * chunks / threads;
                let mut sink = 0_u64;
                for mask in masks {
                    barrier.wait();
                    let selected: Vec<usize> = range.clone().filter(|&index| mask[index]).collect();
                    for (position, &index) in selected.iter().enumerate() {
                        if software_prefetch && let Some(&next) = selected.get(position + 1) {
                            prefetch(buffer[next * size..].as_ptr());
                            prefetch(buffer[next * size + 64..].as_ptr());
                        }
                        sink = sink
                            .wrapping_add(read_chunk(&buffer[index * size..(index + 1) * size]));
                    }
                    barrier.wait();
                }
                std::hint::black_box(sink);
            });
        }
        for _ in masks {
            barrier.wait();
            let started = Instant::now();
            barrier.wait();
            elapsed += started.elapsed().as_secs_f64();
        }
    });
    elapsed
}

fn main() {
    let options = parse();
    let bytes = options.megabytes << 20;
    let buffer: Vec<u8> = (0..bytes).map(|index| (index * 131 + 7) as u8).collect();
    println!(
        "缓冲 {} MiB，{} 线程，保留 {:.0}% 的块，每种设置 {} 遍",
        options.megabytes,
        options.threads,
        options.keep * 100.0,
        options.passes
    );
    println!("块大小\t全读 GB/s\t稀疏/全读 耗时比\t加软件预取\t理想比");
    for &size in &options.chunks {
        let chunks = bytes / size;
        let full = masks(chunks, options.passes, 1.0, 1);
        let sparse = masks(chunks, options.passes, options.keep, size as u64);
        let kept = sparse.iter().flatten().filter(|kept| **kept).count() as f64
            / (chunks * options.passes) as f64;
        run(&buffer, size, options.threads, &full[..2], false);
        let dense = run(&buffer, size, options.threads, &full, false);
        let plain = run(&buffer, size, options.threads, &sparse, false);
        let fetched = run(&buffer, size, options.threads, &sparse, true);
        let rate = (chunks * size * options.passes) as f64 / dense / 1e9;
        println!(
            "{size}\t{rate:.1}\t\t{:.3}\t\t\t{:.3}\t\t{kept:.3}",
            plain / dense,
            fetched / dense
        );
    }
}
