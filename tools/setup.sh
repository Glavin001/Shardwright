#!/usr/bin/env bash
# One-shot, idempotent setup of everything Shardwright needs to build, bake
# and run every oracle. Safe to re-run: finished steps are skipped.
#
#   tools/setup.sh            # everything
#   tools/setup.sh --check    # only report what is installed
#   tools/setup.sh --no-oracles   # Rust build + glTF validator only
#
# Installs (paths overridable through the environment):
#   * system packages (Ubuntu/Debian, when run as root or with sudo):
#     build tools, cmake, Python venv, the X/GL runtime libraries gmsh needs
#   * Rust stable (rustup) if cargo is missing, then `cargo build --release`
#   * Python oracle env $FRACENV (default /opt/fracenv) from
#     tools/harness/requirements.txt (Kratos, gmsh, scikit-fem, CoACD, ...)
#   * flatc v24.3.25 (matches the `flatbuffers` crate pin) at
#     $ORACLES/flatbuffers/build/flatc (default ORACLES=/opt/oracles)
#   * Voro++ (pinned commit) and the test oracle $ORACLES/voro_oracle
#   * Node dependencies of tools/gltf_validate (Khronos glTF Validator)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRACENV="${FRACENV:-/opt/fracenv}"
ORACLES="${ORACLES:-/opt/oracles}"
FLATBUFFERS_TAG="v24.3.25"
VORO_COMMIT="b0dac575a47af0f90b5b100e6dc199a493c7cb83"
MODE="all"
case "${1:-}" in
  --check) MODE="check" ;;
  --no-oracles) MODE="core" ;;
  "") ;;
  *) echo "usage: $0 [--check|--no-oracles]"; exit 2 ;;
esac

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
have() { command -v "$1" > /dev/null 2>&1; }
SUDO=""
if [ "$(id -u)" -ne 0 ] && have sudo; then SUDO="sudo"; fi

check() {
  local ok=0
  st() { if eval "$2" > /dev/null 2>&1; then echo "  [ok]      $1"; else echo "  [missing] $1"; ok=1; fi; }
  echo "Shardwright environment:"
  st "prefracture (release build)" "test -x '$ROOT/target/release/prefracture'"
  if have cargo; then echo "  [ok]      cargo"; else echo "  [info]    cargo not installed (only needed to rebuild)"; fi
  st "node + glTF validator" "have node && test -d '$ROOT/tools/gltf_validate/node_modules/gltf-validator'"
  st "python oracle env ($FRACENV)" "'$FRACENV/bin/python' -c 'import KratosMultiphysics, KratosMultiphysics.StructuralMechanicsApplication, KratosMultiphysics.LinearSolversApplication, gmsh, skfem, coacd, manifold3d, trimesh, scipy'"
  st "flatc $FLATBUFFERS_TAG" "'$ORACLES/flatbuffers/build/flatc' --version | grep -q '${FLATBUFFERS_TAG#v}'"
  st "voro_oracle" "test -x '$ORACLES/voro_oracle'"
  return $ok
}

if [ "$MODE" = "check" ]; then check; exit $?; fi

# ---- system packages -------------------------------------------------------
if have apt-get && { [ "$(id -u)" -eq 0 ] || [ -n "$SUDO" ]; }; then
  pkgs=(build-essential cmake git curl ca-certificates pkg-config python3-venv python3-pip
        libglu1-mesa libgl1 libxrender1 libxcursor1 libxft2 libxinerama1 libgomp1 libfontconfig1)
  missing=()
  for p in "${pkgs[@]}"; do dpkg -s "$p" > /dev/null 2>&1 || missing+=("$p"); done
  if [ ${#missing[@]} -gt 0 ]; then
    say "apt: installing ${missing[*]}"
    $SUDO apt-get update -qq
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${missing[@]}"
  fi
else
  say "skipping system packages (no apt-get or no root); needed: C++ toolchain, cmake, git, python3-venv, GL/X runtime libs for gmsh"
fi

# ---- Rust ------------------------------------------------------------------
if ! have cargo; then
  say "installing Rust (rustup, stable)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi
say "cargo build --release"
(cd "$ROOT" && cargo build --release --workspace)

# ---- Node: Khronos glTF validator -----------------------------------------
if have npm; then
  if [ ! -d "$ROOT/tools/gltf_validate/node_modules/gltf-validator" ]; then
    say "npm ci (tools/gltf_validate)"
    (cd "$ROOT/tools/gltf_validate" && npm ci --no-audit --no-fund)
  fi
else
  say "node/npm not found: the glTF schema gate will report the validator as unavailable"
fi

if [ "$MODE" = "core" ]; then check || true; exit 0; fi

# ---- Python oracle environment ---------------------------------------------
mkdir -p "$(dirname "$FRACENV")" "$ORACLES" 2>/dev/null || $SUDO mkdir -p "$(dirname "$FRACENV")" "$ORACLES"
if [ ! -x "$FRACENV/bin/python" ]; then
  say "python venv $FRACENV"
  python3 -m venv "$FRACENV" 2>/dev/null || { $SUDO python3 -m venv "$FRACENV" && $SUDO chown -R "$(id -u):$(id -g)" "$FRACENV"; }
fi
if ! "$FRACENV/bin/python" -c "import KratosMultiphysics.StructuralMechanicsApplication, KratosMultiphysics.LinearSolversApplication, gmsh, skfem, coacd, manifold3d, trimesh" > /dev/null 2>&1; then
  say "pip install -r tools/harness/requirements.txt"
  "$FRACENV/bin/pip" install -q --upgrade pip
  "$FRACENV/bin/pip" install -q -r "$ROOT/tools/harness/requirements.txt"
fi

# ---- flatc (schema gate) ---------------------------------------------------
if ! "$ORACLES/flatbuffers/build/flatc" --version 2>/dev/null | grep -q "${FLATBUFFERS_TAG#v}"; then
  say "building flatc $FLATBUFFERS_TAG"
  rm -rf "$ORACLES/flatbuffers"
  git clone -q --depth 1 --branch "$FLATBUFFERS_TAG" https://github.com/google/flatbuffers.git "$ORACLES/flatbuffers"
  log="$ORACLES/flatc-build.log"
  if ! { cmake -S "$ORACLES/flatbuffers" -B "$ORACLES/flatbuffers/build" -DCMAKE_BUILD_TYPE=Release \
           -DFLATBUFFERS_BUILD_TESTS=OFF -DFLATBUFFERS_BUILD_FLATLIB=OFF -DFLATBUFFERS_BUILD_FLATHASH=OFF \
         && cmake --build "$ORACLES/flatbuffers/build" --target flatc -j "$(nproc 2>/dev/null || echo 4)"; } > "$log" 2>&1; then
    tail -30 "$log"; echo "flatc build failed (full log: $log)"; exit 1
  fi
fi

# ---- Voro++ differential oracle --------------------------------------------
if [ ! -x "$ORACLES/voro_oracle" ]; then
  say "building Voro++ ($VORO_COMMIT) and voro_oracle"
  if [ ! -f "$ORACLES/voro/src/voro++.cc" ]; then
    rm -rf "$ORACLES/voro"
    git clone -q https://github.com/chr1shr/voro.git "$ORACLES/voro"
    git -C "$ORACLES/voro" checkout -q "$VORO_COMMIT"
  fi
  g++ -O2 -I"$ORACLES/voro/src" "$ROOT/tools/oracles/voro_oracle.cc" "$ORACLES/voro/src/voro++.cc" -o "$ORACLES/voro_oracle"
fi

say "done"
check
