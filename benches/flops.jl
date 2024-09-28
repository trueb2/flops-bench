using BenchmarkTools, LinearAlgebra

const M10 = 10_000_000

function native_dot32(a, b)
    return dot(a, b)
end

function native_dot64(a, b)
    return dot(a, b)
end

function simd_dot32(a, b)
    result = 0.0f0
    @simd for i in eachindex(a, b)
        @inbounds result += a[i] * b[i]
    end
    return result
end

function simd_dot64(a, b)
    result = 0.0
    @simd for i in eachindex(a, b)
        @inbounds result += a[i] * b[i]
    end
    return result
end

function iter_dot32(a, b)
    sum(a[i] * b[i] for i in 1:M10)
end

function iter_dot64(a, b)
    sum(a[i] * b[i] for i in 1:M10)
end

a32 = zeros(Float32, M10)
b32 = zeros(Float32, M10)
a64 = zeros(Float64, M10)
b64 = zeros(Float64, M10)

println("native_dot32")
display(@benchmark native_dot32($a32, $b32))

println("native_dot64")
display(@benchmark native_dot64($a64, $b64))

println("simd_dot32")
display(@benchmark simd_dot32($a32, $b32))

println("simd_dot64")
display(@benchmark simd_dot64($a64, $b64))

println("iter_dot32")
display(@benchmark iter_dot32($a32, $b32))

println("iter_dot64")
display(@benchmark iter_dot64($a64, $b64))
