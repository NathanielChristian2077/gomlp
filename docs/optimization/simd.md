# Float32 e SIMD

A primeira camada de otimização é o kernel de forward denso `float32` em `internal/kernel/f32`. Ele tem fallback escalar e backend `simd/archsimd` habilitado somente com `GOEXPERIMENT=simd` em AMD64.

O kernel ainda não está integrado à MLP. Antes da integração, cada alteração deve passar por:

1. teste de equivalência contra o kernel escalar com tolerância `1e-4`;
2. benchmark isolado do kernel;
3. benchmark end-to-end de inferência;
4. validação de que não ocorreram alocações por forward.

A ordem planejada é: forward denso float32, dispatch SIMD, forward esparso float32, e somente depois treino float32 e vetorização de backpropagation.
