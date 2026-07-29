#!/bin/sh
# Print the newest CUpti_ActivityKernel* struct version that this CUDA toolkit
# provides AND that exposes the four latency-timestamp fields the injection
# library reads (queued / submitted / start / end).
#
# Prints nothing if none compiles, which the Makefile treats as "fall back to
# the CUDA 12.x default and warn".
#
# Inputs (from the Makefile): CC, INCLUDES
#
# NOTE on the negative control below: the first version of this script used
# `mktemp -t probe.XXXXXX.c`, which appends the random suffix AFTER `.c`. The
# resulting filename did not end in `.c`, so clang treated the probe as a
# *linker input*, compiled nothing, and exited 0 -- the probe reported success
# for every candidate even with no cupti.h on the system. A probe that cannot
# fail proves nothing, so the control below asserts that a deliberately broken
# translation unit really does fail before any result is trusted.

set -u

CC="${CC:-cc}"
INCLUDES="${INCLUDES:-}"

tmpdir="${TMPDIR:-/tmp}"
probe="$tmpdir/lumen_cupti_probe.$$.c"
control="$tmpdir/lumen_cupti_control.$$.c"
trap 'rm -f "$probe" "$control"' EXIT INT TERM

# `-x c` forces the language regardless of how the filename is interpreted.
try_compile() {
    # shellcheck disable=SC2086
    $CC $INCLUDES -x c -fsyntax-only "$1" >/dev/null 2>&1
}

# --- negative control: this MUST fail, or the probe is meaningless ---------
cat > "$control" <<'EOF'
#include <this_header_does_not_exist_anywhere.h>
int main(void) { return 0; }
EOF
if try_compile "$control"; then
    echo "probe-struct.sh: compiler accepted a file with a missing header;" >&2
    echo "  the struct probe cannot be trusted. Pass KERNEL_STRUCT= explicitly." >&2
    exit 0
fi

# --- real probe ------------------------------------------------------------
for candidate in \
    CUpti_ActivityKernel9 \
    CUpti_ActivityKernel8 \
    CUpti_ActivityKernel7 \
    CUpti_ActivityKernel6 \
    CUpti_ActivityKernel5 \
    CUpti_ActivityKernel4
do
    cat > "$probe" <<EOF
#include <cupti.h>
int main(void) {
    $candidate k;
    /* All four timestamps must exist: without queued/submitted this tool
     * measures nothing that CUDA events could not already measure. */
    (void)k.queued;
    (void)k.submitted;
    (void)k.start;
    (void)k.end;
    (void)k.name;
    (void)k.correlationId;
    (void)k.deviceId;
    (void)k.streamId;
    (void)k.gridX;
    (void)k.blockX;
    (void)k.dynamicSharedMemory;
    (void)k.staticSharedMemory;
    return 0;
}
EOF
    if try_compile "$probe"; then
        printf '%s\n' "$candidate"
        exit 0
    fi
done

exit 0
