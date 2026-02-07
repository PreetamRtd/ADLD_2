.global _dot_product
.text
.balign 16

_dot_product:
    // x0 = A, x1 = B, x2 = count
    movi    v0.4s, #0           // Accumulator for SIMD
    fmov    s1, wzr             // Scalar accumulator for tail

loop_dot:
    cmp     x2, #4
    blt     tail_dot

    ld1     {v2.4s}, [x0], #16
    ld1     {v3.4s}, [x1], #16

    fmla    v0.4s, v2.4s, v3.4s

    sub     x2, x2, #4
    b       loop_dot

tail_dot:
    // Handle remaining 1-3 elements
    cbz     x2, reduce_dot
    
tail_loop:
    ldr     s2, [x0], #4        // Load single float from A
    ldr     s3, [x1], #4        // Load single float from B
    fmul    s2, s2, s3          // Multiply
    fadd    s1, s1, s2          // Add to scalar accumulator
    
    subs    x2, x2, #1
    bne     tail_loop

reduce_dot:
    // Reduce v0 (4 lanes) to scalar
    faddp   v0.4s, v0.4s, v0.4s // [a+b, c+d, a+b, c+d]
    faddp   v0.4s, v0.4s, v0.4s // [a+b+c+d, ...]
    
    // Add tail accumulator
    fadd    s0, s0, s1
    
    ret