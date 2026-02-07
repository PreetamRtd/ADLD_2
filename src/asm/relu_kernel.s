.global _relu_kernel
.text
.balign 16

_relu_kernel:
    movi    v1.4s, #0

loop_relu:
    cmp     x1, #0
    beq     end_relu

    ld1     {v0.4s}, [x0]
    fmax    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x0]

    add     x0, x0, #16
    sub     x1, x1, #4
    b       loop_relu

end_relu:
    ret