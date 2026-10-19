//! Once a CUDA model is ready, no request allocates device memory: everything a
//! request uses was reserved at load. Recording starts after `preload_weights`
//! returns and covers requests driven through `Session` the way the server drives
//! them (cold prompts of one and several slices, decode greedy and sampled, a
//! generated tail caught up, short and long suffixes, a divergent prompt that
//! restarts, the same prompt again, a prompt longer than the context refused and
//! a decode into the context's last slot). CUPTI records every allocation call,
//! synchronous, stream-ordered, managed or from a pool, the program's own and the
//! CUDA libraries' alike, and every driver call that makes device memory without
//! an allocation record (a module or library load, a function or global lookup,
//! which loads a lazily loaded module's code, a virtual-memory create or map);
//! none may be made by the program, and a window that lost records fails.
//! What a request may still take is the code and static data of a library
//! kernel's module loaded the first time a request launches it (CUDA's default
//! lazy loading): a CUDA library makes those calls itself, so the code that calls
//! the driver lies outside this executable, which tells them from the program's,
//! and the serving plan's allowance covers them. A virtual-memory call fails
//! whoever makes it. Program code another shared object runs, as a callback that
//! ends in a tail call to the driver, would be taken for that object's; Lumen
//! hands no CUDA library a callback. The loads and the static data the driver records are
//! printed, and with
//! `LUMEN_NO_ALLOC_ISOLATED_GPU=1`
//! (nothing else on the GPU) free memory may fall by at most one allocation
//! granule, the driver's smallest unit of device memory, across the requests.
//! Before it is trusted the recorder must see a synchronous driver allocation, a
//! stream allocation and a buffer clone, each as an allocation call, and a module
//! load. `LUMEN_CUDA_PROFILE` creates CUDA events per stage and is not part of a
//! request's contract.
#![cfg(all(feature = "cuda", feature = "test-cupti"))]

mod common;

use common::gdn_hybrid::{
    build_gdn_hybrid_lbc_with, build_qwen35_cpu_lbc, gdn_model_hyperparams_with, Qwen35Fixture,
};
use cudarc::cupti::result::activity;
use cudarc::cupti::sys as cupti;
use cudarc::driver::{result as driver, sys as driver_sys, CudaContext};
use lumen_format::QuantScheme;
use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::config::RuntimeConfig;
use lumen_runtime::cuda::CudaBackend;
use lumen_runtime::kv::KvPrecision;
use lumen_runtime::runtime_defaults;
use lumen_runtime::sampling::SamplingParams;
use lumen_runtime::session::Session;
use lumen_runtime::weight::provider_sync::SyncWeightProvider;
use std::alloc::{alloc, dealloc, Layout};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};

const CONTEXT: usize = 8192;
const BUFFER: usize = 8 << 20;

/// One device allocation CUPTI recorded.
#[derive(Debug)]
#[allow(dead_code)]
struct Allocation {
    bytes: u64,
    kind: cupti::CUpti_ActivityMemoryKind,
    is_async: bool,
    correlation: u32,
    /// The variable's name, for a module's static device memory.
    name: String,
}

impl Allocation {
    /// The driver loading a module's static data, with no allocation call.
    fn is_module_load(&self) -> bool {
        self.correlation == 0
            && self.kind
                == cupti::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC
    }
}

static RECORDED: Mutex<Vec<Allocation>> = Mutex::new(Vec::new());
/// Driver calls that make device memory, or load a module's code, without an
/// allocation record.
const MEMORY_CALLS: [cupti::CUpti_driver_api_trace_cbid; 13] = [
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuModuleLoad,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal_v2,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuLibraryGetModule,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuLibraryGetGlobal,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuMemCreate,
    cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuMemMap,
];

/// The [`MEMORY_CALLS`] the program made while a window records.
static CALLED: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// The loads and lookups a CUDA library made while a window records: its lazy
/// load of a kernel's module.
static LIBRARY_LOADS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static WATCHING: AtomicBool = AtomicBool::new(false);

/// Why a recording window cannot be trusted: a buffer CUPTI could not be given,
/// or a record it could not decode.
static LOST: Mutex<Option<String>> = Mutex::new(None);
static REGISTER: Once = Once::new();

fn layout() -> Layout {
    Layout::from_size_align(BUFFER, 8).unwrap()
}

extern "C" fn buffer_requested(buffer: *mut *mut u8, size: *mut usize, max_records: *mut usize) {
    // SAFETY: CUPTI hands this buffer back to `buffer_completed`, which frees it.
    let ptr = unsafe { alloc(layout()) };
    if ptr.is_null() {
        *LOST.lock().unwrap() = Some("no host memory for a CUPTI buffer".into());
    }
    // SAFETY: CUPTI's out-parameters; a zero size makes it drop the records,
    // which the dropped count then shows.
    unsafe {
        *buffer = ptr;
        *size = if ptr.is_null() { 0 } else { BUFFER };
        *max_records = 0;
    }
}

extern "C" fn buffer_completed(
    _context: driver_sys::CUcontext,
    _stream: u32,
    buffer: *mut u8,
    _size: usize,
    valid: usize,
) {
    let mut record: *mut cupti::CUpti_Activity = std::ptr::null_mut();
    let mut recorded = RECORDED.lock().unwrap();
    loop {
        // SAFETY: `buffer` holds `valid` bytes of activity records CUPTI wrote.
        match unsafe { activity::get_next_record(buffer, valid, &mut record) } {
            Ok(()) => {}
            Err(e) if e.0 == cupti::CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED => break,
            Err(e) => {
                *LOST.lock().unwrap() = Some(format!("a CUPTI record could not be read: {e:?}"));
                break;
            }
        }
        if unsafe { (*record).kind } != cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY2 {
            continue;
        }
        // Every memory record version shares this prefix (kind through isAsync).
        let memory = unsafe { &*(record as *const cupti::CUpti_ActivityMemory3) };
        let operation = memory.memoryOperationType;
        let kind = memory.memoryKind;
        let host = matches!(
            kind,
            cupti::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE
                | cupti::CUpti_ActivityMemoryKind::CUPTI_ACTIVITY_MEMORY_KIND_PINNED
        );
        if operation
            == cupti::CUpti_ActivityMemoryOperationType::CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION
            && !host
        {
            recorded.push(Allocation {
                bytes: memory.bytes,
                kind,
                is_async: memory.isAsync != 0,
                correlation: memory.correlationId,
                name: if memory.name.is_null() {
                    String::new()
                } else {
                    // SAFETY: CUPTI's NUL-terminated name, valid while the record is.
                    unsafe { std::ffi::CStr::from_ptr(memory.name) }
                        .to_string_lossy()
                        .into_owned()
                },
            });
        }
    }
    if !buffer.is_null() {
        // SAFETY: allocated in `buffer_requested` with the same layout.
        unsafe { dealloc(buffer, layout()) };
    }
}

unsafe extern "C" fn driver_call(
    _userdata: *mut std::ffi::c_void,
    domain: cupti::CUpti_CallbackDomain,
    cbid: cupti::CUpti_CallbackId,
    data: *const std::ffi::c_void,
) {
    if !WATCHING.load(Ordering::SeqCst)
        || domain != cupti::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_DRIVER_API
    {
        return;
    }
    // SAFETY: a driver-API callback's data is a `CUpti_CallbackData`.
    let data = unsafe { &*(data as *const cupti::CUpti_CallbackData) };
    if data.callbackSite != cupti::CUpti_ApiCallbackSite::CUPTI_API_ENTER {
        return;
    }
    // SAFETY: CUPTI's NUL-terminated name of the called function.
    let name = unsafe { std::ffi::CStr::from_ptr(data.functionName) }
        .to_string_lossy()
        .into_owned();
    let mapping = [
        cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuMemCreate,
        cupti::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuMemMap,
    ]
    .iter()
    .any(|&c| c as cupti::CUpti_CallbackId == cbid);
    if !mapping && !program_made_call() {
        LIBRARY_LOADS.lock().unwrap().push(name);
    } else {
        CALLED.lock().unwrap().push(name);
    }
}

/// Whether this executable, not a CUDA library, made the driver call a callback
/// is reporting: the first frame below the driver's own lies in this executable.
/// A stack with no frame below the driver counts as the program's.
fn program_made_call() -> bool {
    extern "C" {
        fn backtrace(frames: *mut *mut std::ffi::c_void, size: std::ffi::c_int) -> std::ffi::c_int;
    }
    let object = |addr: *const std::ffi::c_void| {
        // SAFETY: zeroed is a valid `Dl_info`, which dladdr fills.
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        // SAFETY: dladdr only reads the loader's tables; `info` outlives the call.
        let found = unsafe { libc::dladdr(addr, &mut info) } != 0 && !info.dli_fname.is_null();
        found.then(|| {
            // SAFETY: the loader's NUL-terminated path of the object.
            let path = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) };
            (info.dli_fbase, path.to_string_lossy().into_owned())
        })
    };
    let program = object(program_made_call as *const std::ffi::c_void).map(|(base, _)| base);
    let mut frames = [std::ptr::null_mut(); 64];
    // SAFETY: backtrace writes at most `frames.len()` return addresses.
    let n = unsafe { backtrace(frames.as_mut_ptr(), frames.len() as std::ffi::c_int) };
    let mut in_driver = false;
    // A return address is the instruction after the call; one byte back lies
    // inside the calling function.
    for frame in &frames[..n.max(0) as usize] {
        match object(frame.cast::<u8>().wrapping_sub(1).cast()) {
            Some((_, path)) if path.contains("libcuda.so") => in_driver = true,
            Some((base, _)) if in_driver => return Some(base) == program,
            _ => {}
        }
    }
    true
}

/// What a window recorded: device allocations, the [`MEMORY_CALLS`] the program
/// made and the loads a CUDA library made.
struct Recorded {
    allocations: Vec<Allocation>,
    memory_calls: Vec<String>,
    library_loads: Vec<String>,
}

/// Records device allocations and memory calls from `start` until `finish`.
struct Recorder;

impl Recorder {
    fn start() -> Self {
        REGISTER.call_once(|| {
            activity::register_callbacks(Some(buffer_requested), Some(buffer_completed))
                .expect("register the CUPTI activity buffers");
            let mut subscriber: cupti::CUpti_SubscriberHandle = std::ptr::null_mut();
            // SAFETY: `subscriber` is a valid out-parameter; the callback lives
            // as long as the process, and so does the subscription.
            unsafe {
                cudarc::cupti::result::subscribe(
                    &mut subscriber,
                    Some(driver_call),
                    std::ptr::null_mut(),
                )
                .expect("subscribe to CUPTI callbacks");
                for call in MEMORY_CALLS {
                    cudarc::cupti::result::enable_callback(
                        1,
                        subscriber,
                        cupti::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_DRIVER_API,
                        call as cupti::CUpti_CallbackId,
                    )
                    .expect("enable a driver callback");
                }
            }
        });
        RECORDED.lock().unwrap().clear();
        *LOST.lock().unwrap() = None;
        CALLED.lock().unwrap().clear();
        LIBRARY_LOADS.lock().unwrap().clear();
        WATCHING.store(true, Ordering::SeqCst);
        activity::enable(cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY2)
            .expect("enable CUPTI memory records");
        Recorder
    }

    /// What the window recorded, or why it lost records.
    fn finish(self, ctx: &CudaContext) -> Result<Recorded, String> {
        WATCHING.store(false, Ordering::SeqCst);
        activity::disable(cupti::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMORY2).unwrap();
        activity::flush_all(cupti::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED as u32)
            .unwrap();
        if let Some(lost) = LOST.lock().unwrap().take() {
            return Err(lost);
        }
        for context in [std::ptr::null_mut(), ctx.cu_ctx()] {
            let mut dropped = 0;
            // SAFETY: `dropped` is a valid out-parameter.
            unsafe { activity::get_num_dropped_records(context, 0, &mut dropped) }.unwrap();
            if dropped != 0 {
                return Err(format!("CUPTI dropped {dropped} records"));
            }
        }
        Ok(Recorded {
            allocations: std::mem::take(&mut *RECORDED.lock().unwrap()),
            memory_calls: std::mem::take(&mut *CALLED.lock().unwrap()),
            library_loads: std::mem::take(&mut *LIBRARY_LOADS.lock().unwrap()),
        })
    }
}

/// The recorder sees a synchronous driver allocation, a stream allocation, a
/// device-to-device clone and a module load made in the context the backend
/// uses.
fn recorder_sees_every_allocation_kind(ctx: &std::sync::Arc<CudaContext>) {
    let stream = ctx.default_stream();
    let recorder = Recorder::start();
    ctx.bind_to_thread().unwrap();
    // SAFETY: freed at once; nothing reads it.
    unsafe {
        let ptr = driver::malloc_sync(1 << 20).unwrap();
        driver::free_sync(ptr).unwrap();
    }
    let a = stream.alloc_zeros::<u8>(1 << 20).unwrap();
    let b = a.clone();
    stream.synchronize().unwrap();
    drop((a, b));
    let ptx = c".version 7.0\n.target sm_52\n.address_size 64\n";
    // SAFETY: an empty PTX module, unloaded at once.
    unsafe {
        let mut module = std::ptr::null_mut();
        driver_sys::cuModuleLoadData(&mut module, ptx.as_ptr().cast())
            .result()
            .unwrap();
        driver_sys::cuModuleUnload(module).result().unwrap();
    }
    let seen = recorder.finish(ctx).unwrap();
    let calls = seen
        .allocations
        .iter()
        .filter(|a| !a.is_module_load())
        .count();
    assert!(
        calls >= 3,
        "the recorder saw {calls} of 3 planted allocation calls: {:?}",
        seen.allocations
    );
    assert!(
        !seen.memory_calls.is_empty(),
        "the recorder did not see the planted module load"
    );
}

fn provider(lbc: &[u8], label: &str) -> SyncWeightProvider {
    let dir = std::env::temp_dir().join(format!("lumen_no_alloc_{label}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.lbc");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(lbc)
        .unwrap();
    SyncWeightProvider::open(&path).expect("open provider")
}

fn backend(provider: &SyncWeightProvider, precision: KvPrecision) -> CudaBackend {
    let hp = provider.lbc().header.hyperparams;
    // The model defaults the server sets before it builds its backend, so the
    // routes are the ones it serves.
    runtime_defaults::set_path_is_server(true);
    runtime_defaults::set_model_dense_quant(provider.output_proj_quant);
    runtime_defaults::set_model_primary_quant(provider.lbc().header.quantization.scheme);
    runtime_defaults::set_model_block_count(hp.num_layers);
    runtime_defaults::set_model_is_moe(hp.num_experts.unwrap_or(0) > 0);
    let mut cuda = CudaBackend::new(0).expect("a CUDA GPU");
    cuda.set_kv_precision(precision).unwrap();
    cuda.set_global_tensors(
        provider.embedding.clone(),
        provider.final_norm.clone(),
        provider.output_proj.clone(),
    );
    cuda.init(&hp).expect("init");
    cuda.preload_weights(provider).expect("preload");
    cuda
}

fn prompt(len: usize, vocab: usize, seed: usize) -> Vec<u32> {
    (0..len)
        .map(|i| ((i * 7919 + 13 + seed) % vocab) as u32)
        .collect()
}

fn free_memory(ctx: &CudaContext) -> usize {
    ctx.bind_to_thread().unwrap();
    driver::mem_get_info().unwrap().0
}

/// The driver's smallest unit of device memory on this GPU.
fn allocation_granule(ctx: &CudaContext) -> usize {
    ctx.bind_to_thread().unwrap();
    // SAFETY: a zeroed property block with its type and location set is what the
    // query reads; `granule` is a valid out-parameter.
    unsafe {
        let mut prop: driver_sys::CUmemAllocationProp = std::mem::zeroed();
        prop.type_ = driver_sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = driver_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = ctx.ordinal() as i32;
        let mut granule = 0;
        driver_sys::cuMemGetAllocationGranularity(
            &mut granule,
            &prop,
            driver_sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        .result()
        .unwrap();
        granule
    }
}

fn stack_limit(ctx: &CudaContext) -> usize {
    ctx.bind_to_thread().unwrap();
    driver::ctx::get_limit(driver_sys::CUlimit::CU_LIMIT_STACK_SIZE).unwrap()
}

/// The requests of one session, in the order a server sees them.
fn requests(cuda: &CudaBackend, provider: &SyncWeightProvider, precision: KvPrecision) {
    let hp = provider.lbc().header.hyperparams;
    let vocab = hp.vocab_size as usize;
    let threshold = Session::DEFAULT_SUFFIX_THRESHOLD;
    let config = RuntimeConfig {
        kv_precision: precision,
        max_seq_len: CONTEXT,
        ..RuntimeConfig::default()
    };
    let greedy = SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let mut s = Session::new_without_host_kv_mirror(config, hp, greedy).unwrap();
    let decode = |s: &mut Session, n: usize| {
        for _ in 0..n {
            s.next_token(cuda, provider).expect("decode");
        }
    };

    // A cold one-slice prompt, then decode.
    s.extend_with_cache(&prompt(15, vocab, 1), cuda, provider, threshold)
        .expect("cold prompt");
    decode(&mut s, 4);
    // The generated tail is caught up and a long suffix prefilled after it.
    let mut next = s.tokens().to_vec();
    next.extend(prompt(600, vocab, 2));
    s.extend_with_cache(&next, cuda, provider, threshold)
        .expect("long suffix");
    // A short suffix is decoded token by token.
    next.extend(prompt(threshold / 2, vocab, 3));
    s.extend_with_cache(&next, cuda, provider, threshold)
        .expect("short suffix");
    // Sampled decode reads the logits.
    s.set_sampling(SamplingParams {
        temperature: 0.8,
        seed: Some(7),
        ..SamplingParams::default()
    });
    decode(&mut s, 4);
    // A prompt that diverges early restarts cold, over several slices.
    let mut divergent = next[..10].to_vec();
    divergent.extend(prompt(3000, vocab, 4));
    s.extend_with_cache(&divergent, cuda, provider, threshold)
        .expect("divergent prompt");
    // The same prompt again.
    s.extend_with_cache(&divergent, cuda, provider, threshold)
        .expect("same prompt");
    decode(&mut s, 2);
    // A prompt longer than the context is refused; the next request is served.
    assert!(s
        .extend_with_cache(&prompt(CONTEXT + 1, vocab, 5), cuda, provider, threshold)
        .is_err());
    // The first token comes from the prompt's logits; the second is decoded into
    // the context's last slot.
    s.extend_with_cache(&prompt(CONTEXT - 1, vocab, 6), cuda, provider, threshold)
        .expect("a prompt filling the context");
    decode(&mut s, 2);
}

/// The fixtures: a dense GDN hybrid, a Qwen3.5-shaped MoE model with Q8_0
/// experts, and the same with every scheme the CPU decodes.
fn model(label: &str) -> Vec<u8> {
    match label {
        "gdn-hybrid" => build_gdn_hybrid_lbc_with(gdn_model_hyperparams_with(CONTEXT as u32)),
        "qwen35" | "qwen35-quant" => build_qwen35_cpu_lbc(
            &Qwen35Fixture {
                mixed_quant: label == "qwen35-quant",
                expert_quant: Some(QuantScheme::Q8_0),
                routed_experts: Some(64),
                ..Qwen35Fixture::default()
            },
            CONTEXT as u32,
        ),
        other => panic!("no fixture {other}"),
    }
}

const PRECISIONS: [(&str, KvPrecision); 3] = [
    ("f32", KvPrecision::F32),
    ("bf16", KvPrecision::Bf16),
    ("f16", KvPrecision::F16),
];

/// One model at one KV precision, in this process.
fn check(label: &str, precision: KvPrecision) -> Result<(), String> {
    let provider = provider(&model(label), label);
    let ctx = CudaContext::new(0).unwrap();
    recorder_sees_every_allocation_kind(&ctx);
    let cuda = backend(&provider, precision);
    let free = free_memory(&ctx);
    let stack = stack_limit(&ctx);
    let recorder = Recorder::start();
    requests(&cuda, &provider, precision);
    let recorded = recorder.finish(&ctx)?;
    let (loads, calls): (Vec<_>, Vec<_>) = recorded
        .allocations
        .into_iter()
        .partition(Allocation::is_module_load);
    let free_after = free_memory(&ctx);
    println!(
        "{label}/{precision:?}: {} allocation calls, {} library loads and lookups, static data {loads:?}, free {free} -> {free_after} bytes",
        calls.len(),
        recorded.library_loads.len()
    );
    if !calls.is_empty() {
        return Err(format!("requests allocated device memory: {calls:?}"));
    }
    if !recorded.memory_calls.is_empty() {
        return Err(format!(
            "requests made device memory outside the allocation records: {:?}",
            recorded.memory_calls
        ));
    }
    if stack_limit(&ctx) != stack {
        return Err("a request grew the stack".into());
    }
    let granule = allocation_granule(&ctx);
    if std::env::var("LUMEN_NO_ALLOC_ISOLATED_GPU").as_deref() == Ok("1")
        && free.saturating_sub(free_after) > granule
    {
        return Err(format!(
            "free memory fell from {free} to {free_after} bytes, more than one {granule}-byte granule"
        ));
    }
    Ok(())
}

/// Every case runs in a process of its own, as a server loads one model per
/// process: a module an earlier backend loaded stays loaded in the context, so
/// in a shared process a later case would not show its own first use.
#[test]
#[ignore = "needs a GPU and the CUPTI library"]
fn no_request_allocates_device_memory_after_ready() {
    const TEST: &str = "no_request_allocates_device_memory_after_ready";
    if let Ok(case) = std::env::var("LUMEN_NO_ALLOC_CASE") {
        let (label, name) = case.split_once('/').expect("<model>/<precision>");
        let precision = PRECISIONS
            .iter()
            .find(|(n, _)| *n == name)
            .expect("precision")
            .1;
        check(label, precision).unwrap_or_else(|e| panic!("{case}: {e}"));
        return;
    }
    let mut failed = Vec::new();
    for label in ["gdn-hybrid", "qwen35", "qwen35-quant"] {
        for (name, _) in PRECISIONS {
            let case = format!("{label}/{name}");
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([TEST, "--exact", "--ignored", "--nocapture"])
                .env("LUMEN_NO_ALLOC_CASE", &case)
                .status()
                .unwrap();
            if !status.success() {
                failed.push(format!("{case} ({status})"));
            }
        }
    }
    assert!(failed.is_empty(), "cases that failed: {failed:?}");
}
