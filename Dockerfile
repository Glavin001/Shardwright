# syntax=docker/dockerfile:1
# Shardwright `prefracture` with every oracle (Kratos FEM, CoACD, flatc,
# Voro++, Khronos glTF validator). One-line use:
#
#   docker build -t shardwright .
#   docker run --rm -v "$PWD/out:/work/out" shardwright \
#       bake --input benchmarks/assets/rc_column.glb --config benchmarks/configs/bake.toml --out /work/out
#
# Behind a TLS-intercepting proxy: put its CA certificate(s) (*.crt) in
# tools/docker/certs/ and pass the proxy, e.g.
#   docker build --network host --build-arg HTTPS_PROXY=$HTTPS_PROXY -t shardwright .

FROM ubuntu:24.04 AS base
ARG DEBIAN_FRONTEND=noninteractive
COPY tools/docker/certs/ /usr/local/share/ca-certificates/extra/
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends ca-certificates curl git python3 python3-venv nodejs \
      libglu1-mesa libgl1 libxrender1 libxcursor1 libxft2 libxinerama1 libgomp1 libfontconfig1 \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    CARGO_HTTP_CAINFO=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

FROM base AS builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends build-essential cmake pkg-config npm python3-pip \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /shardwright
COPY . .
# release binaries without debug info inside the image
ENV CARGO_PROFILE_RELEASE_DEBUG=0 FRACENV=/opt/fracenv ORACLES=/opt/oracles
RUN tools/setup.sh \
 && mkdir -p /opt/oracles-min/flatbuffers/build \
 && cp /opt/oracles/flatbuffers/build/flatc /opt/oracles-min/flatbuffers/build/ \
 && cp /opt/oracles/voro_oracle /opt/oracles-min/

FROM base AS runtime
COPY --from=builder /opt/fracenv /opt/fracenv
COPY --from=builder /opt/oracles-min /opt/oracles
COPY --from=builder /shardwright/target/release/prefracture /shardwright/target/release/prefracture
COPY --from=builder /shardwright/tools /shardwright/tools
COPY --from=builder /shardwright/benchmarks /shardwright/benchmarks
COPY --from=builder /shardwright/crates/frac-material/materials.toml /shardwright/crates/frac-material/materials.toml
ENV FRACENV=/opt/fracenv ORACLES=/opt/oracles PATH=/shardwright/target/release:$PATH
WORKDIR /shardwright
ENTRYPOINT ["prefracture"]
CMD ["--help"]
