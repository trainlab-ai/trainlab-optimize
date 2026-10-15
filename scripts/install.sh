#!/bin/sh
# TrainLab Optimize installer for macOS and Linux.
#
#   curl -LsSf https://trainlab.ai/install.sh | sh
#
# Installs uv (Astral's Python tool manager) if it's missing, then installs trainlab into its
# own isolated environment with its own Python. Nothing else on the machine changes: no system
# Python, no sudo, no packages in your projects. To remove it: uv tool uninstall trainlab-optimize
#
# Options (environment variables):
#   TRAINLAB_PACKAGE   what to install (default trainlab-optimize from PyPI; also a wheel file,
#                      a URL, or git+https://... for pre-releases)
#   TRAINLAB_PYTHON    Python version for trainlab's environment (default 3.12)
set -eu

PKG="${TRAINLAB_PACKAGE:-trainlab-optimize}"
PY="${TRAINLAB_PYTHON:-3.12}"

say() { printf '%s\n' "$*"; }
fail() { printf 'trainlab install: %s\n' "$*" >&2; exit 1; }

UV="$(command -v uv 2>/dev/null || true)"
if [ -z "$UV" ]; then
    for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        [ -x "$c" ] && UV="$c" && break
    done
fi
if [ -z "$UV" ]; then
    say "Installing uv, the Python tool manager trainlab is installed with ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        fail "needs curl or wget to download uv. Install one of them and run this again."
    fi
    for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        [ -x "$c" ] && UV="$c" && break
    done
    [ -n "$UV" ] || fail "installing uv failed (see above). Check the internet connection or proxy, or open a new terminal, and run this again."
fi

say "Installing trainlab-optimize (Python $PY, isolated) ..."
"$UV" tool install --python "$PY" --force "$PKG" || fail "the install failed (see above)."
"$UV" tool update-shell >/dev/null 2>&1 || true

BIN="$("$UV" tool dir --bin 2>/dev/null || printf '%s' "$HOME/.local/bin")"
# Git Bash / MSYS on Windows: uv prints C:\Users\...; the shell's PATH uses /c/Users/...
if command -v cygpath >/dev/null 2>&1; then BIN="$(cygpath -u "$BIN")"; fi
TL="$BIN/trainlab"
[ -x "$TL" ] || TL="$BIN/trainlab.exe"
[ -x "$TL" ] || TL="$(command -v trainlab 2>/dev/null || true)"
[ -n "$TL" ] || fail "installed, but the trainlab command can't be found in $BIN."

say ""
say "Installed $("$TL" --version)."
case ":$PATH:" in
    *":$BIN:"*) ;;
    *) say "Open a new terminal (or run: export PATH=\"$BIN:\$PATH\") so the trainlab command is found." ;;
esac
say ""
say "Next, in your project folder:"
say "  trainlab ui                                  # the app in your browser"
say "  trainlab run . --usage billing.csv --open    # 1. where the money goes"
say "  trainlab trace -- python app.py              # 2. exact cost per line, no code change"
say "  trainlab trace --capture -- python app.py    # 3. then: trainlab experiment run --max-cost 2 --open"
