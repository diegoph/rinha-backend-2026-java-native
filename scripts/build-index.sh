#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REFS="${1:-resources/references.json.gz}"
OUT="${2:-resources/index.bin}"
rm -rf out app.jar sources.txt
find src/main/java -name '*.java' > sources.txt
javac --release 25 -d out @sources.txt
jar --create --file app.jar --main-class rinha.Main -C out .
if [[ -n "${JAVA_BUILD_OPTS:-}" ]]; then
  read -r -a JVM_OPTS <<< "$JAVA_BUILD_OPTS"
else
  JVM_OPTS=(-Xms512m -Xmx1400m -XX:+UseSerialGC -Dkmeans.k=1024 -Dkmeans.sample=65536 -Dkmeans.iters=35 -Dsplit.enabled=true -Dsplit.max=500 -Dsplit.maxParts=10)
fi
java "${JVM_OPTS[@]}" -cp app.jar rinha.BuildSplitKMeansIndex "$REFS" "$OUT"
ls -lh "$OUT"
