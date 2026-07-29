/* Lumen CUPTI activity injection library.
 *
 * Out-of-process per-kernel profiler for the Lumen CUDA decode path. Loaded by
 * the CUDA driver via CUDA_INJECTION64_PATH, so the Lumen binary needs no
 * rebuild, no CUPTI link dependency, and no source change.
 *
 * Why this exists alongside the in-process event profiler
 * ------------------------------------------------------
 * The Rust profiler (LUMEN_CUDA_PROFILE) brackets *source regions* with CUDA
 * events. That answers "which phase owns the time" but it cannot see inside a
 * region, and a bracket span silently includes any idle the GPU spent waiting
 * for the host to submit the next launch.
 *
 * CUPTI latency timestamps expose exactly that missing axis. With
 * cuptiActivityEnableLatencyTimestamps(1), every kernel record carries four
 * timestamps instead of two:
 *
 *   queued    -- the launch was enqueued by the host
 *   submitted -- the driver submitted it to the GPU
 *   start     -- the GPU began executing it
 *   end       -- the GPU finished it
 *
 * from which the two quantities a decode campaign actually needs fall out
 * per kernel: submit latency (submitted - queued), launch-to-start latency
 * (start - submitted), and true device busy time (end - start). Summing
 * (end - start) over a token gives real busy time; comparing it against the
 * token's wall time gives the honest idle fraction that event brackets hide.
 *
 * nsys is the usual tool for this, but it cannot finalize inside the Modal
 * container ("No GPU associated to the given UUID"). CUPTI works because it
 * runs in-process and never needs to resolve a device UUID out of band.
 *
 * Output
 * ------
 * One CSV row per kernel (and, optionally, per memcpy/memset). Timestamps are
 * raw CUPTI nanoseconds on a single monotonic device timeline, so rows are
 * directly differenceable across records. Sorting by `start` reconstructs the
 * execution order; sorting by `queued` reconstructs the submission order, and
 * the two differing is itself the finding.
 *
 * Environment
 * -----------
 *   LUMEN_CUPTI_CSV      output path (default /tmp/lumen-cupti.csv)
 *   LUMEN_CUPTI_MEMOPS   set to 1 to also record memcpy/memset activity
 *   LUMEN_CUPTI_QUIET    set to 1 to suppress the stderr banner
 *
 * See README.md for the exact run recipe.
 */

#include <cupti.h>

#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CUPTI's kernel activity struct is versioned and grows with each CUDA
 * release. The Makefile probes which version this toolkit provides and passes
 * it in; the fallback targets CUDA 12.x. */
#ifndef LUMEN_CUPTI_KERNEL_STRUCT
#define LUMEN_CUPTI_KERNEL_STRUCT CUpti_ActivityKernel9
#endif
typedef LUMEN_CUPTI_KERNEL_STRUCT lumen_kernel_record_t;

#define LUMEN_STR2(x) #x
#define LUMEN_STR(x) LUMEN_STR2(x)
#define LUMEN_KERNEL_STRUCT_NAME LUMEN_STR(LUMEN_CUPTI_KERNEL_STRUCT)

/* CUPTI wants 8-byte-aligned buffers. 8 MiB holds a few tens of thousands of
 * records, so a 128-token decode run flushes only a handful of times. */
#define BUF_SIZE (8u * 1024u * 1024u)
#define BUF_ALIGN 8u
#define ALIGN_UP(p, a) (((uintptr_t)(p) + ((a) - 1)) & ~(uintptr_t)((a) - 1))

static FILE *g_csv;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_rows;
static uint64_t g_dropped;
static int g_want_memops;
static int g_quiet;
static int g_finalized;

static void note(const char *fmt, ...) {
  if (g_quiet) {
    return;
  }
  va_list ap;
  va_start(ap, fmt);
  fputs("[CUPTI] ", stderr);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
}

static void check(CUptiResult res, const char *what) {
  if (res == CUPTI_SUCCESS) {
    return;
  }
  const char *msg = NULL;
  cuptiGetResultString(res, &msg);
  note("WARN %s failed: %s (%d)", what, msg ? msg : "?", (int)res);
}

/* CSV escaping: kernel names can in principle contain a comma or quote. */
static void write_quoted(FILE *f, const char *s) {
  fputc('"', f);
  for (; s && *s; ++s) {
    if (*s == '"') {
      fputc('"', f);
    }
    fputc(*s, f);
  }
  fputc('"', f);
}

/* CUPTI reports an unavailable timestamp as 0. Latency timestamps in
 * particular are 0 unless cuptiActivityEnableLatencyTimestamps succeeded, so
 * emitting them verbatim (rather than computing a difference here) keeps the
 * "not measured" case distinguishable from a genuine zero interval. */
static void emit_kernel(const lumen_kernel_record_t *k) {
  fputs("kernel,", g_csv);
  write_quoted(g_csv, k->name ? k->name : "?");
  fprintf(g_csv,
          ",%" PRIu32 ",%" PRIu32 ",%" PRIu32
          ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
          ",%" PRId32 ",%" PRId32 ",%" PRId32
          ",%" PRId32 ",%" PRId32 ",%" PRId32
          ",%" PRId32 ",%" PRId32 "\n",
          k->deviceId, k->streamId, k->correlationId,
          (uint64_t)k->queued, (uint64_t)k->submitted,
          (uint64_t)k->start, (uint64_t)k->end,
          k->gridX, k->gridY, k->gridZ,
          k->blockX, k->blockY, k->blockZ,
          k->dynamicSharedMemory, k->staticSharedMemory);
  g_rows++;
}

static void emit_memcpy(const CUpti_ActivityMemcpy5 *m) {
  fputs("memcpy,", g_csv);
  write_quoted(g_csv, "memcpy");
  fprintf(g_csv,
          ",%" PRIu32 ",%" PRIu32 ",%" PRIu32
          ",0,0,%" PRIu64 ",%" PRIu64
          ",%" PRIu64 ",%u,0,0,0,0,0,0\n",
          m->deviceId, m->streamId, m->correlationId,
          (uint64_t)m->start, (uint64_t)m->end,
          (uint64_t)m->bytes, (unsigned)m->copyKind);
  g_rows++;
}

static void emit_memset(const CUpti_ActivityMemset4 *m) {
  fputs("memset,", g_csv);
  write_quoted(g_csv, "memset");
  fprintf(g_csv,
          ",%" PRIu32 ",%" PRIu32 ",%" PRIu32
          ",0,0,%" PRIu64 ",%" PRIu64
          ",%" PRIu64 ",0,0,0,0,0,0,0\n",
          m->deviceId, m->streamId, m->correlationId,
          (uint64_t)m->start, (uint64_t)m->end,
          (uint64_t)m->bytes);
  g_rows++;
}

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                                      size_t *max_num_records) {
  uint8_t *raw = (uint8_t *)malloc(BUF_SIZE + BUF_ALIGN);
  if (!raw) {
    *buffer = NULL;
    *size = 0;
    *max_num_records = 0;
    return;
  }
  *buffer = (uint8_t *)ALIGN_UP(raw, BUF_ALIGN);
  *size = BUF_SIZE;
  *max_num_records = 0; /* fill the buffer completely before handing it back */
}

/* Called on a CUPTI worker thread, so serialize CSV writes. */
static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t stream_id,
                                      uint8_t *buffer, size_t size,
                                      size_t valid_size) {
  (void)ctx;
  (void)stream_id;
  (void)size;

  pthread_mutex_lock(&g_lock);
  if (g_csv && valid_size > 0) {
    CUpti_Activity *rec = NULL;
    for (;;) {
      CUptiResult res = cuptiActivityGetNextRecord(buffer, valid_size, &rec);
      if (res == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      }
      if (res != CUPTI_SUCCESS) {
        check(res, "cuptiActivityGetNextRecord");
        break;
      }
      switch (rec->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
          emit_kernel((const lumen_kernel_record_t *)rec);
          break;
        case CUPTI_ACTIVITY_KIND_MEMCPY:
          emit_memcpy((const CUpti_ActivityMemcpy5 *)rec);
          break;
        case CUPTI_ACTIVITY_KIND_MEMSET:
          emit_memset((const CUpti_ActivityMemset4 *)rec);
          break;
        default:
          break;
      }
    }
  }

  size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(NULL, 0, &dropped) == CUPTI_SUCCESS &&
      dropped > 0) {
    /* Dropped records mean the table below is INCOMPLETE. Surface it rather
     * than letting a short table read as a fast run. */
    g_dropped += (uint64_t)dropped;
  }
  pthread_mutex_unlock(&g_lock);

  free(buffer);
}

static void finalize(void) {
  if (g_finalized) {
    return;
  }
  g_finalized = 1;

  check(cuptiActivityFlushAll(1), "cuptiActivityFlushAll");

  pthread_mutex_lock(&g_lock);
  if (g_csv) {
    fflush(g_csv);
    if (g_csv != stderr) {
      fclose(g_csv);
    }
    g_csv = NULL;
  }
  uint64_t rows = g_rows;
  uint64_t dropped = g_dropped;
  pthread_mutex_unlock(&g_lock);

  if (dropped > 0) {
    fprintf(stderr,
            "[CUPTI] WARN %" PRIu64 " activity records were DROPPED -- the CSV "
            "is incomplete; do not sum it as if it were whole\n",
            dropped);
  }
  note("wrote %" PRIu64 " rows", rows);
}

/* Entry point the CUDA driver calls when CUDA_INJECTION64_PATH names this
 * library. Runs before CUDA is initialized, which is exactly when activity
 * kinds must be enabled to catch the first launch. */
int InitializeInjection(void) {
  const char *path = getenv("LUMEN_CUPTI_CSV");
  const char *memops = getenv("LUMEN_CUPTI_MEMOPS");
  const char *quiet = getenv("LUMEN_CUPTI_QUIET");

  g_want_memops = (memops && memops[0] == '1');
  g_quiet = (quiet && quiet[0] == '1');
  if (!path || !path[0]) {
    path = "/tmp/lumen-cupti.csv";
  }

  g_csv = fopen(path, "w");
  if (!g_csv) {
    fprintf(stderr, "[CUPTI] ERROR cannot open %s for writing; falling back to stderr\n",
            path);
    g_csv = stderr;
  }

  fputs("kind,name,device,stream,correlation,queued_ns,submitted_ns,start_ns,end_ns,"
        "grid_x,grid_y,grid_z,block_x,block_y,block_z,dyn_smem,static_smem\n",
        g_csv);

  /* Latency timestamps must be enabled BEFORE the activity kinds, and they are
   * what populate `queued` and `submitted`. Without them those columns are 0
   * and only device busy time (end - start) is measurable. */
  check(cuptiActivityEnableLatencyTimestamps(1),
        "cuptiActivityEnableLatencyTimestamps");

  check(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed),
        "cuptiActivityRegisterCallbacks");

  /* CONCURRENT_KERNEL is the one to use: plain KERNEL serializes kernel
   * execution to time it, which would change the very overlap we are here to
   * measure. */
  check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
        "cuptiActivityEnable(CONCURRENT_KERNEL)");

  if (g_want_memops) {
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY),
          "cuptiActivityEnable(MEMCPY)");
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET),
          "cuptiActivityEnable(MEMSET)");
  }

  atexit(finalize);

  note("injected: csv=%s memops=%d struct=%s", path, g_want_memops,
       LUMEN_KERNEL_STRUCT_NAME);
  return 1;
}
