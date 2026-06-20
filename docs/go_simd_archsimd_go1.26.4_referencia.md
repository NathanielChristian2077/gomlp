# Go SIMD — Referência de Engenharia
## `simd/archsimd` no Go 1.26.4 (amd64)

> **Escopo.** Este arquivo consolida, em português, a documentação relevante do pacote experimental oficial `simd/archsimd` do Go 1.26.4, com foco em uso correto, tipos vetoriais, máscaras, detecção de CPU, padrões de código e aplicação em kernels de MLP.
>
> Não é uma cópia literal da página oficial inteira. A API experimental contém milhares de combinações de métodos por tipo; por isso, esta referência organiza a superfície completa por famílias e preserva links/commands para consultar a assinatura exata de qualquer identificador na toolchain instalada.
>
> **Versão-alvo:** Go 1.26.4  
> **Arquitetura documentada:** `amd64`  
> **Pacote:** `simd/archsimd`  
> **Status:** experimental, fora da promessa de compatibilidade Go 1.

---

## 1. O que o pacote é

`simd/archsimd` fornece operações SIMD específicas de arquitetura. No Go 1.26, ele está disponível para AMD64 e expõe vetores de 128, 256 e 512 bits. Os vetores correspondem a registradores de hardware e a maior parte das operações é reconhecida pelo compilador como intrínseco de uma instrução de máquina.

A consequência prática é simples:

- `Float32x4` representa 4 `float32` em 128 bits.
- `Float32x8` representa 8 `float32` em 256 bits.
- `Float32x16` representa 16 `float32` em 512 bits.
- `Float64x2`, `Float64x4`, `Float64x8` seguem a mesma lógica para `float64`.
- Inteiros e máscaras seguem a mesma ideia, variando pelo tamanho da lane.

O pacote só existe quando o experimento é habilitado:

```bash
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go run ./cmd/...
GOEXPERIMENT=simd go test -bench=. ./...
```

Sem `GOEXPERIMENT=simd`, o import falha. Isso não é bug. É a API dizendo “estou aqui, mas ainda não assinei contrato”.

---

## 2. Regras de uso que realmente importam

1. **Não exponha tipos SIMD em APIs públicas.**  
   O pacote é específico de arquitetura e experimental. Mantenha `archsimd.Float32x8` escondido em `internal/` ou em implementações privadas.

2. **Use vetores como valores locais.**  
   Não tome endereço de vetor, não coloque vetor em heap, não guarde vetor em struct de longa vida e não faça coleção de vetores. Isso prejudica a alocação em registradores.

3. **Cheque suporte da CPU antes de chamar um kernel específico.**  
   O pacote fornece `archsimd.X86` para isso.

4. **Faça fallback escalar.**  
   SIMD é um backend. A semântica do programa não pode depender de SIMD existir.

5. **Benchmark antes de celebrar.**  
   Vetorizar um loop que é limitado por memória, branches ou alocação pode não trazer ganho. A CPU não entrega performance por mérito moral.

6. **Aceite pequenas diferenças numéricas.**  
   `MulAdd` usa FMA, que arredonda uma vez. A versão escalar costuma multiplicar e depois somar, arredondando em etapas diferentes. Compare `float32` com tolerância, não com igualdade exata.

---

## 3. Detecção de recursos da CPU

O pacote expõe:

```go
var archsimd.X86 archsimd.X86Features
```

Os métodos relevantes:

```go
archsimd.X86.AVX()      // AVX disponível?
archsimd.X86.AVX2()     // AVX2 disponível? implica AVX.
archsimd.X86.AVX512()   // conjunto básico AVX-512 disponível? implica AVX e AVX2.
archsimd.X86.FMA()      // fused multiply-add disponível? implica AVX.
```

Há ainda consultas para extensões especializadas:

```text
AVX512BITALG
AVX512GFNI
AVX512VAES
AVX512VBMI
AVX512VBMI2
AVX512VNNI
AVX512VPCLMULQDQ
AVX512VPOPCNTDQ
AVXAES
AVXVNNI
SHA
VAES
```

### Dispatch recomendado para a MLP

Para a primeira versão do kernel `float32`:

```go
func canUseF32x8FMA() bool {
    return archsimd.X86.AVX2() && archsimd.X86.FMA()
}
```

E no dispatcher:

```go
func denseForwardF32(input, weights, bias, output []float32, in, out int) {
    if canUseF32x8FMA() && out >= 8 {
        denseForwardF32x8(input, weights, bias, output, in, out)
        return
    }
    denseForwardScalarF32(input, weights, bias, output, in, out)
}
```

Não use `Float32x16` só porque ele parece mais heroico: ele exige AVX-512. Em CPUs sem AVX-512, isso não é “mais lento”, é uma instrução indisponível.

---

## 4. Larguras vetoriais e tipos principais

### 4.1 Ponto flutuante

| Tipo | Lanes | Largura | Uso natural |
|---|---:|---:|---|
| `Float32x4` | 4 | 128 bits | blocos pequenos, compatibilidade mais ampla |
| `Float32x8` | 8 | 256 bits | principal alvo AVX2 para MLP `float32` |
| `Float32x16` | 16 | 512 bits | AVX-512 |
| `Float64x2` | 2 | 128 bits | `float64`, bloco pequeno |
| `Float64x4` | 4 | 256 bits | AVX/AVX2 com `float64` |
| `Float64x8` | 8 | 512 bits | AVX-512 |

### 4.2 Inteiros com sinal

| Elemento | 128 bits | 256 bits | 512 bits |
|---|---|---|---|
| `int8` | `Int8x16` | `Int8x32` | `Int8x64` |
| `int16` | `Int16x8` | `Int16x16` | `Int16x32` |
| `int32` | `Int32x4` | `Int32x8` | `Int32x16` |
| `int64` | `Int64x2` | `Int64x4` | `Int64x8` |

### 4.3 Inteiros sem sinal

| Elemento | 128 bits | 256 bits | 512 bits |
|---|---|---|---|
| `uint8` | `Uint8x16` | `Uint8x32` | `Uint8x64` |
| `uint16` | `Uint16x8` | `Uint16x16` | `Uint16x32` |
| `uint32` | `Uint32x4` | `Uint32x8` | `Uint32x16` |
| `uint64` | `Uint64x2` | `Uint64x4` | `Uint64x8` |

### 4.4 Máscaras

As máscaras acompanham a largura da lane, por exemplo:

```text
Mask8x16,  Mask8x32,  Mask8x64
Mask16x8,  Mask16x16, Mask16x32
Mask32x4,  Mask32x8,  Mask32x16
Mask64x2,  Mask64x4,  Mask64x8
```

Uma comparação como:

```go
mask := x.Greater(zero)
```

produz uma máscara, não um vetor de `bool`. Depois você a usa em operações como:

```go
masked := x.Masked(mask)
merged := x.Merge(other, mask)
```

Para ReLU, a lógica vetorial conceitual é:

```text
mask = x > 0
relu = x mascarado por mask
```

Mas isso só entra depois que o GEMV denso estiver estável. Não tente simultaneamente aprender SIMD, máscara, ReLU e DSA, a menos que queira transformar a branch em museu de regressões.

---

## 5. Anatomia de uma família vetorial

Quase todos os tipos vetoriais seguem a mesma estrutura:

### Construção / carregamento

```go
archsimd.BroadcastFloat32x8(x float32) Float32x8
archsimd.LoadFloat32x8(y *[8]float32) Float32x8
archsimd.LoadFloat32x8Slice(s []float32) Float32x8
archsimd.LoadFloat32x8SlicePart(s []float32) Float32x8
archsimd.LoadMaskedFloat32x8(y *[8]float32, mask Mask32x8) Float32x8
```

- `Broadcast...`: replica um escalar em todas as lanes.
- `Load...`: carrega array de tamanho exato.
- `Load...Slice`: exige slice com pelo menos o número completo de lanes.
- `Load...SlicePart`: carrega o que existir e completa as lanes restantes com zero.
- `LoadMasked...`: carrega apenas lanes habilitadas por máscara.

### Armazenamento

```go
v.Store(y *[8]float32)
v.StoreSlice(s []float32)
v.StoreSlicePart(s []float32)
v.StoreMasked(y *[8]float32, mask Mask32x8)
```

- `StoreSlice`: exige espaço para todas as lanes.
- `StoreSlicePart`: grava somente o que couber.

### Aritmética usual

```go
x.Add(y)
x.Sub(y)
x.Mul(y)
x.Div(y)
x.MulAdd(y, z) // (x*y)+z com FMA, quando suportado
x.Min(y)
x.Max(y)
x.Sqrt()
```

### Comparação e máscara

```go
x.Equal(y)
x.NotEqual(y)
x.Greater(y)
x.GreaterEqual(y)
x.Less(y)
x.LessEqual(y)
x.IsNaN()
```

### Transformações e reorganização

```go
x.AsInt32x8()
x.ConvertToInt32()
x.Permute(indices)
x.ConcatPermute(y, indices)
x.GetLo()
x.GetHi()
x.SetLo(y)
x.SetHi(y)
x.GetElem(index)
x.SetElem(index, value)
```

### Operações de compactação relevantes para DSA

Em vetores AVX-512, `Compress` e `Expand` são especialmente importantes:

```go
packed := x.Compress(mask)
expanded := packed.Expand(mask)
```

`Compress` reúne lanes selecionadas para as posições inferiores. Isso conversa diretamente com ativação esparsa, mas exige AVX-512 no caso das variantes `Float32x16` e `Float64x8`. Para a primeira fase da MLP em AVX2, não conte com isso.

---

## 6. `Float32x8`: a família que interessa para a primeira MLP otimizada

`Float32x8` é um vetor de 256 bits com oito `float32`. Ele é o alvo natural para uma MLP `float32` em AMD64 com AVX2.

### Construção correta

```go
xVec := archsimd.BroadcastFloat32x8(x)

wVec := archsimd.LoadFloat32x8Slice(weights[offset:])
outVec := archsimd.LoadFloat32x8Slice(output[offset:])
```

### Acumulação correta

Com FMA disponível:

```go
outVec = xVec.MulAdd(wVec, outVec)
```

Sem FMA:

```go
outVec = outVec.Add(xVec.Mul(wVec))
```

### Armazenamento correto

```go
outVec.StoreSlice(output[offset:])
```

### Observação crucial

O padrão abaixo é **correto**:

```go
archsimd.LoadFloat32x8Slice(output[o:])
```

O padrão abaixo é **errado para essa API**:

```go
archsimd.LoadFloat32x8(&output[o]) // não é a assinatura do pacote
```

A API oferece versões para array fixo e para slice. Use as versões `...Slice` em slices dinâmicos do kernel.

---

## 7. Kernel MLP: `denseForwardF32x8`

### Contrato matemático

Para pesos em layout input-major:

```text
weights[i*out + o]
```

calculamos:

```text
output[o] = bias[o] + Σ_i input[i] * weights[i*out + o]
```

### Implementação de referência

```go
package kernel

import "simd/archsimd"

const lanesF32x8 = 8

func denseForwardF32x8(
    input []float32,
    weights []float32,
    bias []float32,
    output []float32,
    in int,
    out int,
) {
    copy(output[:out], bias[:out])

    vectorEnd := out &^ (lanesF32x8 - 1)
    useFMA := archsimd.X86.FMA()

    for i := 0; i < in; i++ {
        x := input[i]
        xVec := archsimd.BroadcastFloat32x8(x)
        weightRow := i * out

        for o := 0; o < vectorEnd; o += lanesF32x8 {
            acc := archsimd.LoadFloat32x8Slice(output[o:])
            w := archsimd.LoadFloat32x8Slice(weights[weightRow+o:])

            if useFMA {
                acc = xVec.MulAdd(w, acc)
            } else {
                acc = acc.Add(xVec.Mul(w))
            }

            acc.StoreSlice(output[o:])
        }

        // Cauda para out não múltiplo de 8.
        for o := vectorEnd; o < out; o++ {
            output[o] += x * weights[weightRow+o]
        }
    }
}
```

### O que esse kernel exige

- `len(input) >= in`
- `len(weights) >= in*out`
- `len(bias) >= out`
- `len(output) >= out`
- `out` pode ser qualquer valor; a cauda escalar trata o resto.
- O dispatcher só chama esta versão se `X86.AVX2()` for verdadeiro.

### Por que a ordem dos loops é assim

```go
for i := 0; i < in; i++ {
    for o := 0; o < vectorEnd; o += 8 {
        ...
    }
}
```

Cada `input[i]` escala oito pesos contíguos de uma vez. O layout `weights[i*out+o]` deixa esses pesos adjacentes na memória. É exatamente o tipo de acesso que ajuda cache e SIMD a não se odiarem.

### Versão escalar equivalente

```go
func denseForwardScalarF32(
    input []float32,
    weights []float32,
    bias []float32,
    output []float32,
    in int,
    out int,
) {
    copy(output[:out], bias[:out])

    for i := 0; i < in; i++ {
        x := input[i]
        weightRow := i * out

        for o := 0; o < out; o++ {
            output[o] += x * weights[weightRow+o]
        }
    }
}
```

---

## 8. Teste de equivalência escalar vs SIMD

```go
func TestDenseForwardF32x8MatchesScalar(t *testing.T) {
    const (
        in  = 17
        out = 13 // 8 SIMD + 5 de cauda
    )

    input := make([]float32, in)
    weights := make([]float32, in*out)
    bias := make([]float32, out)
    want := make([]float32, out)
    got := make([]float32, out)

    for i := 0; i < in; i++ {
        input[i] = float32(i+1) * 0.07
    }
    for i := 0; i < len(weights); i++ {
        weights[i] = float32((i%11)-5) * 0.03
    }
    for i := 0; i < out; i++ {
        bias[i] = float32(i) * 0.01
    }

    denseForwardScalarF32(input, weights, bias, want, in, out)
    denseForwardF32x8(input, weights, bias, got, in, out)

    const tolerance = 1e-4
    for i := 0; i < out; i++ {
        diff := want[i] - got[i]
        if diff < 0 {
            diff = -diff
        }
        if diff > tolerance {
            t.Fatalf("output[%d]: scalar=%g simd=%g diff=%g",
                i, want[i], got[i], diff)
        }
    }
}
```

A tolerância é necessária porque FMA pode alterar a ordem de arredondamento. O contrato é equivalência numérica prática, não igualdade bit-a-bit.

---

## 9. SIMD e a cabeça `softmax2`

Uma saída de dois logits tem `out=2`.

Isso significa:

```text
vectorEnd = 2 &^ 7 = 0
```

Portanto a cabeça final `... -> 2` fica totalmente no caminho escalar. Isso está certo e não é um problema: o custo importante está nas camadas escondidas largas, como `4096 -> 32`, `32 -> 64` e `64 -> 512`.

Para inferência por classe, não é necessário aplicar softmax apenas para escolher a maior classe:

```go
func classFromLogits(logits []float32) int {
    if logits[1] >= logits[0] {
        return 1
    }
    return 0
}
```

Softmax é necessário para probabilidades e cross-entropy, não para `argmax`.

---

## 10. AVX, AVX2, FMA e AVX-512

### AVX

Geralmente associado a vetores de 256 bits para ponto flutuante. Várias operações `Float32x8` e `Float64x4` exigem AVX.

### AVX2

Amplia suporte e oferece operações inteiras de 256 bits. No pacote, vários construtores e operações em vetores de 256 bits indicam AVX2 como requisito.

### FMA

`MulAdd` usa fused multiply-add:

```text
a*b + c
```

com uma operação combinada. Para a MLP, é a instrução que interessa porque o GEMV é quase todo uma sequência de multiplicar e acumular.

### AVX-512

Entrega 512 bits, por exemplo `Float32x16` e `Float64x8`, e permite operações como `Compress`/`Expand` úteis para máscaras/compactação. Mas é outra classe de compatibilidade de CPU. Não trate AVX-512 como extensão automaticamente disponível em uma máquina AVX2.

---

## 11. Máscaras e ReLU vetorial

A forma conceitual da ReLU é:

```text
relu(x) = max(x, 0)
```

Com `archsimd`, há dois caminhos:

### Caminho simples

```go
zero := archsimd.BroadcastFloat32x8(0)
relu := x.Max(zero)
```

### Caminho explícito com máscara

```go
zero := archsimd.BroadcastFloat32x8(0)
mask := x.Greater(zero)
relu := x.Masked(mask)
```

O primeiro é mais direto para uma ReLU densa. O segundo é mais útil quando a máscara precisa ser reutilizada para estatísticas de ativação, DSA ou seleção condicional.

Não introduza `Compress` no caminho AVX2 esperando que ele apareça do nada. A documentação associa `Compress` em vetores de 512 bits a AVX-512.

---

## 12. DSA e SIMD

A DSA atual trabalha com:

```text
indices ativos + valores ativos
```

A versão SIMD mantém a ideia, mas vetorializa a dimensão de saída:

```go
for k := 0; k < activeN; k++ {
    i := activeIdx[k]
    xVec := archsimd.BroadcastFloat32x8(activeVal[k])
    weightRow := i * out

    for o := 0; o < vectorEnd; o += 8 {
        acc := archsimd.LoadFloat32x8Slice(output[o:])
        w := archsimd.LoadFloat32x8Slice(weights[weightRow+o:])
        acc = xVec.MulAdd(w, acc)
        acc.StoreSlice(output[o:])
    }
}
```

A diferença para o dense é só o loop externo:

```text
dense:  i = 0 até in-1
sparse: k = 0 até activeN-1, com i = activeIdx[k]
```

A parte vetorial é idêntica. Isso é bom: uma única abstração de bloco SIMD atende os dois kernels.

---

## 13. `ClearAVXUpperBits`

O pacote fornece:

```go
archsimd.ClearAVXUpperBits()
```

Ele emite `VZEROUPPER`, limpando bits superiores de registradores Y/Z. O objetivo é reduzir penalidades de transição entre AVX e SSE, causadas por dependências falsas.

Para a primeira versão da MLP em Go puro, **não chame isso aleatoriamente dentro do kernel**. Só investigue se:

- o profiler/assembly mostrar mistura de AVX e SSE;
- houver chamada para código legado/SSE;
- o benchmark demonstrar diferença;
- você estiver saindo deliberadamente de um bloco AVX antes de interoperar com outro caminho.

A própria documentação observa que o compilador pode automatizar isso no futuro.

---

## 14. Operações que merecem atenção especial

### `MulAdd`

Para GEMV/MLP:

```go
acc = x.MulAdd(w, acc)
```

É a operação central.

### `Load...SlicePart` e `StoreSlicePart`

Podem eliminar um tail escalar:

```go
tail := archsimd.LoadFloat32x8SlicePart(weights[offset:])
```

Mas não presuma que isso é mais rápido que uma cauda escalar. Para uma MLP comum, o tail é pequeno e previsível. Primeiro implemente a cauda escalar. Depois benchmark.

### `GetElem` / `SetElem`

Úteis para depuração ou casos raros. Não use em hot loop. Extrair lane por lane mata o motivo de ter SIMD.

### `Permute`, `ConcatPermute`, `Select...`

São úteis para reorganização, redução e kernels especializados. Para GEMV básico, não são necessários.

### `Reciprocal` e `ReciprocalSqrt`

Produzem aproximações. Não use como substituto automático de divisão/sqrt em loss, normalização ou softmax sem medir erro e efeito no treino.

### `Compress` / `Expand`

Muito atraentes para DSA, mas dependem de AVX-512 nas formas de ponto flutuante relevantes. São um tópico de fase posterior, não uma condição para começar AVX2.

---

## 15. API por família: mapa completo de operações

A superfície exata varia por tipo, largura e recurso da CPU. Em vez de listar milhares de repetições quase idênticas, use este mapa.

### Ponto flutuante (`Float32x*`, `Float64x*`)

| Família | Operações típicas |
|---|---|
| Construção | `Broadcast*`, `Load*`, `Load*Slice`, `Load*SlicePart`, `LoadMasked*` |
| Memória | `Store`, `StoreSlice`, `StoreSlicePart`, `StoreMasked` |
| Aritmética | `Add`, `Sub`, `Mul`, `Div`, `MulAdd`, `Min`, `Max`, `Sqrt` |
| Comparação | `Equal`, `NotEqual`, `Greater`, `GreaterEqual`, `Less`, `LessEqual`, `IsNaN` |
| Conversão | `ConvertTo...`, `As...` |
| Máscara | `Masked`, `Merge`, `Compress`, `Expand` quando suportado |
| Reorganização | `GetLo`, `GetHi`, `SetLo`, `SetHi`, `Permute`, `ConcatPermute`, `Select...` |
| Arredondamento | `Ceil`, `Floor`, `Trunc`, `RoundToEven` e variantes `Scaled` |
| Aproximações | `Reciprocal`, `ReciprocalSqrt` |
| Utilitários | `Len`, `String`, `GetElem`, `SetElem` |

### Inteiros (`Int*`, `Uint*`)

Além do equivalente acima, aparecem grupos como:

| Família | Operações típicas |
|---|---|
| Bitwise | `And`, `AndNot`, `Or`, `Xor`, `Not` |
| Shift/rotate | `ShiftLeft`, `ShiftRight`, `ShiftAllLeft`, `ShiftAllRight`, `Rotate...` |
| Saturação | `AddSaturated`, `SubSaturated`, `SaturateTo...`, `TruncateTo...` |
| Popcount | `OnesCount` |
| Packing/unpacking | `Extend...`, `Truncate...`, conversões e reinterpret casts |
| Criptografia | AES/Galois-field em tipos específicos de `uint8` |
| Produto escalar especializado | `DotProduct...` em certas larguras/tipos |

### Máscaras (`Mask*`)

| Operação | Finalidade |
|---|---|
| `Mask...FromBits` | cria máscara de um bitmask |
| `ToBits` | extrai bitmask |
| `And`, `Or` | combina predicados |
| `ToInt...` | converte para vetor inteiro correspondente |

---

## 16. Comandos de consulta local

Use a toolchain exata que você vai compilar:

```bash
GOEXPERIMENT=simd go doc simd/archsimd
GOEXPERIMENT=simd go doc simd/archsimd.Float32x8
GOEXPERIMENT=simd go doc simd/archsimd.Float32x8.MulAdd
GOEXPERIMENT=simd go doc simd/archsimd.LoadFloat32x8Slice
GOEXPERIMENT=simd go doc simd/archsimd.X86Features
GOEXPERIMENT=simd go doc simd/archsimd.X86Features.AVX2
```

Para conferir a toolchain:

```bash
go version
go env GOARCH GOOS GOEXPERIMENT
```

Para visualizar assembly/otimizações em investigação posterior:

```bash
GOEXPERIMENT=simd go test -gcflags='-S' ./internal/...
GOEXPERIMENT=simd go test -bench=. -benchmem ./internal/...
```

Não comece lendo assembly antes de o teste de equivalência passar. Isso é como abrir o motor para ajustar a mistura antes de descobrir que esqueceu de colocar combustível.

---

## 17. Estrutura recomendada na branch `GOptimize`

```text
internal/
  f32/
    dense_scalar.go
    dense_simd_amd64.go
    dense_dispatch_amd64.go
    dense_dispatch_other.go
    relu_scalar.go
    relu_simd_amd64.go
    sparse_scalar.go
    sparse_simd_amd64.go
    kernel_test.go
    benchmark_test.go
```

Sugestão de build tags:

```go
//go:build amd64
```

Para arquivos que importam `simd/archsimd`, a toolchain ainda precisa receber:

```bash
GOEXPERIMENT=simd
```

O fallback escalar continua sendo a implementação universal e de referência.

---

## 18. Checklist para a primeira implementação

- [ ] Criar kernel escalar `float32`.
- [ ] Criar teste escalar vs baseline `float64` com tolerância apropriada.
- [ ] Implementar `denseForwardF32x8`.
- [ ] Implementar dispatcher AVX2/FMA -> SIMD; demais casos -> escalar.
- [ ] Testar `out` múltiplo de 8 e com cauda.
- [ ] Medir `go test -bench` antes de integrar na MLP completa.
- [ ] Integrar apenas forward denso.
- [ ] Só depois vetorizar ReLU.
- [ ] Só depois vetorizar DSA.
- [ ] Só depois avaliar quantização/índices compactos.

---

## 19. Fontes oficiais

1. Go 1.26 Release Notes  
   https://go.dev/doc/go1.26

2. Pacote `simd/archsimd`, versão Go 1.26.4  
   https://pkg.go.dev/simd/archsimd@go1.26.4?GOOS=linux

3. Página atual do pacote  
   https://pkg.go.dev/simd/archsimd

4. Proposta oficial do pacote  
   https://go.dev/issue/73787

5. Guia oficial do assembler Go  
   https://go.dev/doc/asm

6. Código-fonte do pacote na árvore oficial do Go  
   https://cs.opensource.google/go/go/+/go1.26.4:src/simd/archsimd/

---

## 20. Resumo para a nossa MLP

A primeira implementação industrial razoável é:

```text
dados/pesos/ativações: float32
kernel prioritário: Float32x8
largura prática: 8 lanes
dispatch: AVX2 + FMA
fallback: escalar
layout dos pesos: input-major
teste: SIMD ~= scalar com tolerância
benchmark: camada isolada antes de MLP completa
```

A operação central é:

```go
acc = xVec.MulAdd(weightVec, acc)
```

com:

```go
xVec      := archsimd.BroadcastFloat32x8(input[i])
weightVec := archsimd.LoadFloat32x8Slice(weights[row+o:])
acc       := archsimd.LoadFloat32x8Slice(output[o:])
acc       = xVec.MulAdd(weightVec, acc)
acc.StoreSlice(output[o:])
```

Isso é o ponto de partida. O resto é engenharia, benchmark e a inevitável fase em que descobrimos que o cache tinha opiniões próprias.
