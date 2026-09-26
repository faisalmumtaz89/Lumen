// Prints the cuBLASLt ABI facts Lumen's runtime binding (crates/lumen-runtime/src/cuda/cublaslt.rs)
// relies on, from cublasLt.h and, for payload widths, from the library itself: every attribute's width
// is the byte count its GetAttribute reports as needed (a zero-size query); for the algorithm
// configuration that is asked of a real heuristic algorithm of an NVFP4 and of an FP8 plan (the two
// must agree), then read back at that size. Its output is cublaslt_abi.txt beside it.
//
// Needs cuBLASLt 12.8 or newer and a GPU its heuristics offer block-scaled FP4 algorithms for (the
// committed file was printed on compute capability 12.0). With CUDA the toolkit root, e.g.
// /usr/local/cuda:
//   cc -O1 -I$CUDA/include generate_cublaslt_abi.c -L$CUDA/lib64 -lcublasLt -o generate_cublaslt_abi
//   ./generate_cublaslt_abi > cublaslt_abi.txt
#include <cublasLt.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define P(key, v) printf("%s %llu\n", key, (unsigned long long)(v))
#define CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "%s failed: %d\n", #x, (int)s_); exit(1); } } while (0)

static cublasLtHandle_t lt;

static size_t desc_size(cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t a) {
  size_t need = 0;
  CHECK(cublasLtMatmulDescGetAttribute(d, a, NULL, 0, &need));
  return need;
}

static size_t pref_size(cublasLtMatmulPreference_t p, cublasLtMatmulPreferenceAttributes_t a) {
  size_t need = 0;
  CHECK(cublasLtMatmulPreferenceGetAttribute(p, a, NULL, 0, &need));
  return need;
}

// The first heuristic algorithm of the plan `D[m][n] = X[m][k] * W[n][k]^T` in `type`.
static cublasLtMatmulAlgo_t first_algo(cudaDataType_t type, int block_scaled) {
  const uint64_t n = 5120, k = 5120, m = 128;
  cublasLtMatmulDesc_t d;
  CHECK(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
  CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
  CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
  if (block_scaled) {
    cublasLtMatmulMatrixScale_t s = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &s, sizeof s));
    CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &s, sizeof s));
    // The block-scaled heuristic query is refused while the scale pointers are unset. It does not
    // read them, so an aligned address serves (no GEMM runs here).
    const void* sp = (const void*)(uintptr_t)0x7f0000000000ull;
    CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sp, sizeof sp));
    CHECK(cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sp, sizeof sp));
  }
  cublasLtMatrixLayout_t a, b, c;
  CHECK(cublasLtMatrixLayoutCreate(&a, type, k, n, k));
  CHECK(cublasLtMatrixLayoutCreate(&b, type, k, m, k));
  CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, n));
  cublasLtMatmulPreference_t p;
  CHECK(cublasLtMatmulPreferenceCreate(&p));
  uint64_t ws = 32u << 20;
  CHECK(cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  cublasLtMatmulHeuristicResult_t r;
  int got = 0;
  CHECK(cublasLtMatmulAlgoGetHeuristic(lt, d, a, b, c, c, p, 1, &r, &got));
  if (got < 1) { fprintf(stderr, "no heuristic algorithm for type %d\n", (int)type); exit(1); }
  cublasLtMatmulPreferenceDestroy(p);
  cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c);
  cublasLtMatmulDescDestroy(d);
  return r.algo;
}

int main(void) {
  CHECK(cublasLtCreate(&lt));
  printf("# cuBLASLt ABI facts, printed by generate_cublaslt_abi.c from the cublasLt.h and the library\n");
  printf("# whose versions follow: type sizes, alignments and field offsets, enum constants, and\n");
  printf("# attribute value sizes. crates/lumen-runtime/src/cuda/cublaslt.rs checks every value it binds.\n");
  P("header_version", CUBLAS_VER_MAJOR * 10000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH);
  P("library_version", cublasLtGetVersion());

  P("sizeof.cublasLtMatmulAlgo_t", sizeof(cublasLtMatmulAlgo_t));
  P("alignof.cublasLtMatmulAlgo_t", alignof(cublasLtMatmulAlgo_t));
  P("sizeof.cublasLtMatmulHeuristicResult_t", sizeof(cublasLtMatmulHeuristicResult_t));
  P("alignof.cublasLtMatmulHeuristicResult_t", alignof(cublasLtMatmulHeuristicResult_t));
  P("offsetof.cublasLtMatmulHeuristicResult_t.algo", offsetof(cublasLtMatmulHeuristicResult_t, algo));
  P("offsetof.cublasLtMatmulHeuristicResult_t.workspaceSize",
    offsetof(cublasLtMatmulHeuristicResult_t, workspaceSize));
  P("offsetof.cublasLtMatmulHeuristicResult_t.state", offsetof(cublasLtMatmulHeuristicResult_t, state));
  P("offsetof.cublasLtMatmulHeuristicResult_t.wavesCount",
    offsetof(cublasLtMatmulHeuristicResult_t, wavesCount));
  P("offsetof.cublasLtMatmulHeuristicResult_t.reserved", offsetof(cublasLtMatmulHeuristicResult_t, reserved));
  P("sizeof.cublasStatus_t", sizeof(cublasStatus_t));
  P("sizeof.cublasComputeType_t", sizeof(cublasComputeType_t));
  P("sizeof.cudaDataType_t", sizeof(cudaDataType_t));
  P("sizeof.cublasOperation_t", sizeof(cublasOperation_t));

  P("const.CUBLAS_COMPUTE_32F", CUBLAS_COMPUTE_32F);
  P("const.CUDA_R_32F", CUDA_R_32F);
  P("const.CUDA_R_16BF", CUDA_R_16BF);
  P("const.CUDA_R_8F_E4M3", CUDA_R_8F_E4M3);
  P("const.CUDA_R_8F_UE4M3", CUDA_R_8F_UE4M3);
  P("const.CUDA_R_4F_E2M1", CUDA_R_4F_E2M1);
  P("const.CUBLAS_OP_N", CUBLAS_OP_N);
  P("const.CUBLAS_OP_T", CUBLAS_OP_T);
  P("const.CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3", CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);
  P("const.CUBLASLT_REDUCTION_SCHEME_NONE", CUBLASLT_REDUCTION_SCHEME_NONE);
  P("const.CUBLASLT_REDUCTION_SCHEME_INPLACE", CUBLASLT_REDUCTION_SCHEME_INPLACE);
  P("const.CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE", CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE);
  P("const.CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE", CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE);
  P("const.CUBLASLT_MATMUL_DESC_FAST_ACCUM", CUBLASLT_MATMUL_DESC_FAST_ACCUM);

  cublasLtMatmulDesc_t d;
  CHECK(cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
#define DESC(name) do { P("const.CUBLASLT_MATMUL_DESC_" #name, CUBLASLT_MATMUL_DESC_##name); \
    P("size.CUBLASLT_MATMUL_DESC_" #name, desc_size(d, CUBLASLT_MATMUL_DESC_##name)); } while (0)
  DESC(TRANSA);
  DESC(TRANSB);
  DESC(A_SCALE_POINTER);
  DESC(B_SCALE_POINTER);
  DESC(A_SCALE_MODE);
  DESC(B_SCALE_MODE);
  cublasLtMatmulDescDestroy(d);

  cublasLtMatmulPreference_t p;
  CHECK(cublasLtMatmulPreferenceCreate(&p));
#define PREF(name) do { P("const.CUBLASLT_MATMUL_PREF_" #name, CUBLASLT_MATMUL_PREF_##name); \
    P("size.CUBLASLT_MATMUL_PREF_" #name, pref_size(p, CUBLASLT_MATMUL_PREF_##name)); } while (0)
  PREF(MAX_WORKSPACE_BYTES);
  PREF(REDUCTION_SCHEME_MASK);
  cublasLtMatmulPreferenceDestroy(p);

  cublasLtMatmulAlgo_t fp4 = first_algo(CUDA_R_4F_E2M1, 1), fp8 = first_algo(CUDA_R_8F_E4M3, 0);
#define CFG(name) do { size_t w4 = 0, w8 = 0, got = 0; unsigned char buf[64]; \
    CHECK(cublasLtMatmulAlgoConfigGetAttribute(&fp4, CUBLASLT_ALGO_CONFIG_##name, NULL, 0, &w4)); \
    CHECK(cublasLtMatmulAlgoConfigGetAttribute(&fp8, CUBLASLT_ALGO_CONFIG_##name, NULL, 0, &w8)); \
    if (w4 != w8 || w4 > sizeof buf) { fprintf(stderr, #name " needs %zu and %zu bytes\n", w4, w8); exit(1); } \
    CHECK(cublasLtMatmulAlgoConfigGetAttribute(&fp4, CUBLASLT_ALGO_CONFIG_##name, buf, w4, &got)); \
    if (got != w4) { fprintf(stderr, #name " wrote %zu of %zu bytes\n", got, w4); exit(1); } \
    P("const.CUBLASLT_ALGO_CONFIG_" #name, CUBLASLT_ALGO_CONFIG_##name); \
    P("size.CUBLASLT_ALGO_CONFIG_" #name, w4); } while (0)
  CFG(ID);
  CFG(TILE_ID);
  CFG(SPLITK_NUM);
  CFG(REDUCTION_SCHEME);
  CFG(CTA_SWIZZLING);
  CFG(CUSTOM_OPTION);
  CFG(STAGES_ID);
  CFG(INNER_SHAPE_ID);
  CFG(CLUSTER_SHAPE_ID);
  CHECK(cublasLtDestroy(lt));
  return 0;
}
