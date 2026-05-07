# rinha-java-final

Solução Java 25/GraalVM Native para a Rinha de Backend 2026.

## Arquitetura

A aplicação sobe três containers:

- `lb`: HAProxy escutando na porta `9999`.
- `api1`: API Java Native usando Unix Domain Socket em `/sockets/api1.sock`.
- `api2`: API Java Native usando Unix Domain Socket em `/sockets/api2.sock`.

Endpoints expostos pelo load balancer:

- `GET /ready`
- `POST /fraud-score`

## Algoritmo

A solução usa um índice IVF aproximado com reparo exato:

1. O arquivo `references.json.gz` é convertido para vetores inteiros de 14 dimensões, escalados por `10000`.
2. O build executa K-Means determinístico com `k=1024`.
3. Clusters grandes são subdivididos por split de cauda para reduzir p99.
4. O runtime usa `nprobe=1`, escaneando primeiro o cluster mais próximo.
5. O `bbox repair` avalia a caixa mínima/máxima dos demais clusters e escaneia apenas clusters que ainda podem conter vizinhos melhores.
6. O top-5 é ordenado por distância e, em empate, pelo índice original do vetor de referência.
7. A resposta é calculada a partir da quantidade de fraudes no top-5.

A configuração padrão busca manter `0%` de erro com baixa latência p99.

## Otimizações principais

- GraalVM Native Image.
- Unix Domain Socket entre HAProxy e APIs.
- Respostas HTTP pré-montadas.
- Parser manual do payload.
- Índice row-major para o scanner quente.
- Early-exit por dimensão durante o cálculo de distância.
- Fast path para `nprobe=1` sem arrays temporários de cluster.

## Gerar índice

```bash
./scripts/build-index.sh resources/references.json.gz resources/index.bin
```

Parâmetros padrão do índice:

```text
kmeans.k=1024
kmeans.sample=65536
kmeans.iters=35
split.enabled=true
split.max=500
split.maxParts=10
```

## Rodar localmente

```bash
docker compose down -v
docker compose up --build -d
```

## Rodar k6 oficial

Em outro diretório, rode o teste oficial da Rinha:

```bash
cd ~/Downloads/rinha-de-backend-2026
k6 run test/test.js
cat test/results.json
```

## Configuração final

Runtime:

```text
ivf.nprobe=1
knn.slots=1
server.inbuf=768
server.outbuf=512
server.header.max=2048
server.body.max=8192
startup.warmup.searches=0
```

Recursos:

```text
lb   0.12 CPU / 20MB
api1 0.44 CPU / 165MB
api2 0.44 CPU / 165MB
```