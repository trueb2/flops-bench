//! Divan benchmark to compare GFLOPS with native Rust chunks and ndarray BLAS.
extern crate blas_src;

use divan::Bencher;
use ndarray::Array;

fn main() {
    divan::main();
}

const M10: usize = 10_000_000;

#[divan::bench]
fn ndarray_dot32(b: Bencher) {
    b.with_inputs(|| (Array::from_vec(vec![0f32; M10]), Array::from_vec(vec![0f32; M10])))
        .bench_values(|(a, b)| {
            a.dot(&b)
        });
}

#[divan::bench]
fn ndarray_dot64(b: Bencher) {
    b.with_inputs(|| (Array::from_vec(vec![0f64; M10]), Array::from_vec(vec![0f64; M10])))
        .bench_values(|(a, b)| {
            a.dot(&b)
        });
}


#[divan::bench]
fn chunks_dot32(b: Bencher) {
    b.with_inputs(|| (vec![0f32; M10], vec![0f32; M10]))
        .bench_values(|(a, b)| {
            a.chunks_exact(32)
                .zip(b.chunks_exact(32))
                .map(|(a, b)| a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>())
                .sum::<f32>()
        });
}


#[divan::bench]
fn chunks_dot64(b: Bencher) {
    b.with_inputs(|| (vec![0f64; M10], vec![0f64; M10]))
        .bench_values(|(a, b)| {
            a.chunks_exact(32)
                .zip(b.chunks_exact(32))
                .map(|(a, b)| a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f64>())
                .sum::<f64>()
        });
}


#[divan::bench]
fn iter_dot32(b: Bencher) {
    b.with_inputs(|| (vec![0f32; M10], vec![0f32; M10]))
        .bench_values(|(a, b)| {
            a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>()
        });
}


#[divan::bench]
fn iter_dot64(b: Bencher) {
    b.with_inputs(|| (vec![0f64; M10], vec![0f64; M10]))
        .bench_values(|(a, b)| {
            a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f64>()
        });
}
