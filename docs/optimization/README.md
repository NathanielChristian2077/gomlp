# Otimização

A trilha de otimização vive em `optimization-clean`, derivada da baseline atual de `main`. Ela contém contratos de Adam, `float32`, kernels e SIMD sem carregar artefatos da busca GPU ou experimentos de DSA.

Antes de promover uma otimização para `main`, valide correção numérica, ausência de alocações no kernel e benchmark end-to-end.
