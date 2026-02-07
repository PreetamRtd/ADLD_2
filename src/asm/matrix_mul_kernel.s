.global _matrix_mul_kernel
.text
.balign 16

_matrix_mul_kernel:
    stp     x19, x20, [sp, #-48]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]

    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3

    mov     x4, #0

loop_i:
    cmp     x4, x22
    bge     end_i

    mul     x23, x4, x22
    lsl     x23, x23, #2
    add     x23, x19, x23

    mul     x24, x4, x22
    lsl     x24, x24, #2
    add     x24, x21, x24

    mov     x5, #0

loop_j:
    cmp     x5, x22
    bge     end_j

    movi    v0.4s, #0

    mov     x6, #0
    
    mov     x10, x23
    
    lsl     x11, x5, #2
    add     x11, x20, x11

    lsl     x12, x22, #2

loop_k:
    cmp     x6, x22
    bge     end_k

    ld1r    {v1.4s}, [x10], #4

    ld1     {v2.4s}, [x11], x12

    fmla    v0.4s, v2.4s, v1.4s

    add     x6, x6, #1
    b       loop_k

end_k:
    lsl     x13, x5, #2
    add     x13, x24, x13
    st1     {v0.4s}, [x13]

    add     x5, x5, #4
    b       loop_j

end_j:
    add     x4, x4, #1
    b       loop_i

end_i:
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #48
    ret
