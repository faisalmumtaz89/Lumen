//! Single-worker engine wrapper.
//!
//! The Lumen runtime is `Send + Sync` at the trait level but real backends
//! (Metal, CUDA, AMX) hold device handles whose contracts forbid concurrent
//! use across threads. The cheapest correct model is therefore a single
//! worker that owns the engine state and serializes jobs.
//!
//! Handlers send a [`JobRequest`] over an `mpsc` channel. The worker
//! processes one job at a time, sending [`TokenEvent`]s back over a per-job
//! channel. When the job finishes (EOS, max tokens, or cancellation), the
//! worker drops the response sender and the handler observes the channel
//! closing.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use lumen_runtime::compute::ComputeBackend;
use lumen_runtime::engine::SamplingParams;
use lumen_runtime::session::{Session, SuffixPrefillResult};
use lumen_runtime::weight::cache::WeightProvider;
use lumen_runtime::{RuntimeConfig, RuntimeError, ServerMemoryBreakdown};
use lumen_format::ModelHyperparams;

use tokio::sync::mpsc;

use crate::error::ServerError;
use crate::tokenstop::StopMatcher;

/// Per-request channel pool.
///
/// `EngineHandle::submit` used to allocate a fresh
/// `tokio::sync::mpsc::channel::<TokenEvent>(128)` on every HTTP request.
/// Under the 30-min soak (209 requests over 1787 s) this produced
/// a +154.5 MB/h RSS slope rooted in macOS system-allocator fragmentation
///
/// This pool pre-allocates a small set of `(Sender, Receiver)` pairs at
/// worker spawn and recycles them across requests.  `submit` borrows a
/// pair via [`EngineHandle::take_channel_pair`]; the wire-layer
/// [`PooledReceiver`] returns the pair on `Drop`, draining any leftover
/// events first so the next user starts with a clean channel.
///
/// Channel internals are kept alive across requests because the pool
/// retains a `Sender` clone for each pair — the channel never closes for
/// lack of senders, only when the operator stops the server.
///
/// Capacity per pooled channel = [`POOL_CHANNEL_CAPACITY`] = 16 slots, which
/// is more than sufficient at typical decode rates (50 ms/token, so 16
/// in-flight tokens buys 800 ms of write-ahead before backpressure kicks
/// in).  Pool size cap = `inbox_capacity + 1`; overflow allocations are
/// dropped on receiver completion rather than expanding the pool
/// indefinitely.
const POOL_CHANNEL_CAPACITY: usize = 16;

/// Poll interval the worker uses to check the per-job cancellation flag
/// while the per-request token channel is full.
///
/// Under the channel-pool design the pool retains a `Sender` clone for each
/// `(Sender, Receiver)` pair so the channel never closes between requests.
/// That means the worker's `blocking_send` on the token channel cannot
/// detect a vanished `Receiver` via the standard "all senders dropped"
/// signal — the pool's retained `Sender` keeps the channel alive even after
/// the wire layer has dropped the receiver (e.g. on mid-stream client
/// disconnect). A client that opens a streaming POST, receives one chunk,
/// and closes the connection therefore wedges the server permanently —
/// the worker fills the 16-slot channel and then blocks indefinitely on
/// `blocking_send`, freezing the single-worker engine for every subsequent
/// request.
///
/// The remedy is to replace `blocking_send` with a `try_send` loop that
/// polls the per-job `CancellationFlag` between attempts; the wire-layer
/// drive task flips the flag on client disconnect via the
/// `CancellationGuard` it holds for the request's lifetime. A 5 ms poll
/// interval keeps the worker exit latency bounded — subsequent POST
/// requests return 200 within ~1 s — without spinning hot on the pool
/// lock. At 5 ms/iteration the polling overhead is negligible compared to
/// the ~50 ms/token decode budget on the synthetic CPU-naive backend and
/// ~5-20 ms/token on production GPU backends.
///
/// This constant coexists with the `catch_unwind` panic supervisor:
/// if `process_job` panics with the
/// cancel flag still `false`, the supervisor sees the panic, emits a
/// clean `TokenEvent::Error("engine recovered from panic: ...")` to the
/// in-flight reply channel, and on the next iteration of the worker
/// loop the wire layer's `PooledReceiver` drop runs anyway (the body
/// task observed the error and unwound), flipping the cancel flag for
/// any leftover sender clone.  This guarantees both classes of failure
/// (client-side disconnect, server-side panic) leave the client in a
/// clean "stream closed" state rather than wedged on `recv().await`.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Cancellation flag shared between handler and worker.
///
/// `submit()` allocates an `Arc<AtomicBool>` per job, hands one clone to
/// the worker via `WorkerJob`, and returns the other clone wrapped in a
/// [`CancellationGuard`] that the wire layer drops on stream completion
/// or client disconnect.  Drop flips the flag from `false` to `true`; the
/// worker's `send_event_polling_cancel` loop observes the flip on its
/// next poll and aborts the job cleanly without producing more tokens.
///
/// `Ordering::Relaxed` is sufficient — the only invariant is "if the
/// guard's Drop has run, the worker's NEXT load_some sees `true`".
/// There is no other memory we are synchronising on; the worker does not
/// read or write any other state guarded by this flag.
pub type CancellationFlag = Arc<AtomicBool>;

/// panic-supervisor budget for the worker decode loop.
///
/// `process_job` calls into CUDA / Metal kernels that, on saturation
/// OOM, can panic via `cudarc-0.19.x` `unwrap()` on `DriverError`
/// (`CUDA_ERROR_OUT_OF_MEMORY`). Previously such a panic killed the
/// worker thread; the inbox half closed; every subsequent `submit`
/// returned `EngineUnavailable("worker channel closed")`. The failure
/// surfaces under sustained per-job GPU OOMs from saturating device
/// memory (e.g. heavy MoE Q4 server concurrency).
///
/// The fix wraps `process_job` in `catch_unwind`, emits a clean
/// 503-shaped `TokenEvent::Error` to the in-flight job's reply
/// channel, rebuilds the per-worker `Session` (cheap — weights are
/// shared `Arc`s; only KV scratch is reallocated), and continues the
/// loop.  A sustained panic storm (e.g. a configuration bug that
/// guarantees OOM on every job) would otherwise loop forever, so a
/// rolling window caps the budget: `MAX_PANICS_IN_WINDOW` panics
/// inside `PANIC_WINDOW`, beyond which the worker drains the inbox
/// with `engine unhealthy: too many panics` errors and exits.  The
/// HTTP layer continues to serve `/v1/models` and other engine-free
/// endpoints with structured 503 envelopes; operator intervention
/// (process restart) is then the documented path.
///
/// Both constants are env-overridable so production operators can
/// tune the recovery aggressiveness (e.g. a more permissive
/// `LUMEN_SERVER_PANIC_MAX=10` for transient bursty OOMs, or a tighter
/// `LUMEN_SERVER_PANIC_WINDOW_SECS=10` for fail-fast deployments) AND
/// so tests can verify the budget-exhaustion drain without waiting
/// the default 60s window.
const DEFAULT_PANIC_WINDOW_SECS: u64 = 60;
const DEFAULT_MAX_PANICS_IN_WINDOW: usize = 3;

fn panic_window() -> Duration {
    Duration::from_secs(
        std::env::var("LUMEN_SERVER_PANIC_WINDOW_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_PANIC_WINDOW_SECS),
    )
}

fn max_panics_in_window() -> usize {
    std::env::var("LUMEN_SERVER_PANIC_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_PANICS_IN_WINDOW)
}

/// Configuration for the disk-persistent KV cache.
///
/// When supplied to [`EngineWorker::spawn_with_disk_cache`], the worker:
/// - purges stale `.tmp.<pid>` writers at startup,
/// - runs one eviction sweep so the directory fits `budget_bytes`,
/// - re-runs the eviction sweep after each completed job (idempotent and
///   cheap; the typical decision is"no work to do".
///
/// The full save/load round-trip (write the in-memory cache on job exit,
/// load on prefix match at job start) requires deeper Session integration
/// and lands in a follow-up patch. The on-disk module's
/// `disk_save_load_resume_is_bitwise_identical` property test in
/// `lumen-runtime::kv::disk::tests` is the binding correctness contract
/// for that follow-up.
#[derive(Debug, Clone)]
pub struct DiskKvConfig {
    /// Directory the cache files live in.
    pub dir: PathBuf,
    /// Soft byte budget; eviction kicks in when the directory exceeds this.
    /// Set to `u64::MAX` (or `0`) for "no eviction, only orphan purging".
    pub budget_bytes: u64,
}

/// Token-stream events the worker publishes to a request handler.
#[derive(Debug, Clone)]
pub enum TokenEvent {
    /// The worker accepted the job; prefill is done. Useful for clients
    /// that want to display "thinking…" until the first token.
    PrefillDone {
        reused_prefix_len: usize,
        suffix_len: usize,
        prefill_time: Duration,
    },
    /// A single decoded token, both as id and as the incremental decoded
    /// UTF-8 fragment (may be empty if the byte boundary did not yet
    /// resolve to a complete character).
    Token {
        token_id: u32,
        delta_text: String,
    },
    /// Generation ended cleanly. `finish_reason` is one of
    /// `"stop"` (EOS or stop sequence), `"length"` (max tokens reached),
    /// or `"tool_calls"` (the model emitted at least one tool call).
    Done {
        finish_reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    /// Generation failed. The handler should translate this to an HTTP
    /// error response or an SSE `error:` event.
    Error(String),
}

/// Why a generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Natural end of turn: the model emitted an EOS / role-end token.
    Stop,
    /// Hit the answer-token budget (`max_tokens`) or the KV context window.
    Length,
    /// A tool-call block was emitted (and not interrupted).
    ToolCalls,
    /// A caller-supplied textual stop sequence (OpenAI `stop` /
    /// Anthropic `stop_sequences`) was matched in the decoded answer text.
    /// Distinct from `Stop` so Anthropic can report the spec-correct
    /// `stop_reason: "stop_sequence"` (OpenAI collapses both to `"stop"`).
    StopSequence,
}

impl FinishReason {
    /// String form for OpenAI / Anthropic finish_reason fields.
    pub fn as_openai(&self) -> &'static str {
        match self {
            // OpenAI's ChatCompletion schema has a single "stop" value covering
            // both natural EOS and a matched `stop` string.
            Self::Stop | Self::StopSequence => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
        }
    }

    /// String form for Anthropic stop_reason fields.
    pub fn as_anthropic(&self) -> &'static str {
        match self {
            Self::Stop => "end_turn",
            Self::Length => "max_tokens",
            Self::ToolCalls => "tool_use",
            // Anthropic distinguishes a stop-sequence hit from a natural turn end.
            Self::StopSequence => "stop_sequence",
        }
    }
}

/// One job submitted to the engine worker.
#[derive(Debug)]
pub struct JobRequest {
    /// Token ids comprising the FULL prompt (post-tokenization, post-
    /// chat-template). The worker decides whether this extends, diverges
    /// from, or replaces the prior session.
    pub prompt_tokens: Vec<u32>,

    /// Maximum tokens to generate.
    pub max_tokens: usize,

    /// Stop sequences (decoded text); generation halts when the most
    /// recent emitted text ends with any of these. Token-id stop matching
    /// is handled separately via `eos_token_ids`.
    pub stop_text: Vec<String>,

    /// Token ids that count as EOS (model EOS plus any role markers like
    /// `<|im_end|>`).
    pub eos_token_ids: Vec<u32>,

    /// Sampling parameters. The runtime currently honors `temperature` and
    /// `seed`; other fields are forward-compatible no-ops.
    pub sampling: SamplingParams,

    /// Suffix-prefill threshold below which the worker uses single-token
    /// decode rather than dispatching the batched prefill kernel. Set to
    /// 0 to always use batched prefill; default 32.
    pub suffix_threshold: usize,

    /// Whether the prompt opened a `<think>` reasoning block (resolved via
    /// `runtime_defaults::resolve_enable_thinking`). Carried for the Part-4
    /// decode-loop reasoning-budget / forced-close work; the current decode
    /// loop does not yet read it. Default `false` (no reasoning).
    pub enable_thinking: bool,

    /// Maximum reasoning ("thinking") tokens before the decode loop force-
    /// closes the `<think>` block (Part 4). Separate from `max_tokens` (the
    /// answer budget) so the answer is never starved. `0` = unbounded /
    /// unused (Part-4-pending). Carried now so Part 4 can wire enforcement
    /// without touching every constructor again.
    pub reasoning_budget: usize,
}

/// The reply channel the worker uses for token events.
///
/// this is now a [`PooledReceiver`] rather than a raw
/// `mpsc::Receiver<TokenEvent>`.  The handler-facing API is unchanged
/// (`.recv().await` returns `Option<TokenEvent>`); the wrapper exists so
/// the channel pair can return to [`EngineHandle::channel_pool`] on drop
/// rather than being deallocated.
pub type JobResponseChannel = PooledReceiver;

/// Internal message shape: a job plus its reply senders.
struct WorkerJob {
    request: JobRequest,
    tokens_tx: mpsc::Sender<TokenEvent>,
    /// Per-job cancellation flag. The worker checks this
    /// between token emissions; the wire-layer's [`CancellationGuard`]
    /// flips it from `false` to `true` on Drop (stream completion or
    /// client disconnect).  See [`CANCEL_POLL_INTERVAL`] for polling
    /// cadence and [`EngineWorker::send_event_polling_cancel`] for the
    /// observation site.
    cancel: CancellationFlag,
}

/// RAII guard that cancels its associated job on Drop.
///
/// `EngineHandle::submit` returns the guard wrapped inside the
/// [`JobResponseChannel`] (see [`PooledReceiver::cancel_guard`]) so the
/// wire-layer task that drives the SSE body owns the guard for the
/// request lifetime.  When the body task returns — for any reason:
/// normal end-of-stream, client disconnect, panic — the
/// `PooledReceiver` drops, which drops the guard, which sets the
/// cancellation flag, which the worker observes within
/// [`CANCEL_POLL_INTERVAL`].
///
/// Construct via [`CancellationGuard::new`]; the only public surface is
/// `Drop`.  We intentionally do NOT expose a `cancel()` method — the
/// guard's contract is "cancel happens on drop" and exposing manual
/// cancellation would invite callers to forget the drop semantics.
#[derive(Debug)]
pub struct CancellationGuard {
    flag: CancellationFlag,
}

impl CancellationGuard {
    /// Wrap a `CancellationFlag` so the flag flips to `true` when the
    /// guard is dropped.
    pub(crate) fn new(flag: CancellationFlag) -> Self {
        Self { flag }
    }
}

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        // Relaxed is sufficient: the only invariant is "if drop has run,
        // the worker's next load sees true".  There's no other memory
        // ordering required — the worker doesn't read any state that
        // this guard synchronises with.
        self.flag.store(true, Ordering::Relaxed);
    }
}

/// Shared channel pool — bounded `VecDeque` of recycled `(Sender, Receiver)`
/// pairs.  Pre-populated at worker spawn time; `submit` pops a pair (or
/// allocates a fresh one if empty); [`PooledReceiver::drop`] pushes the
/// pair back if the pool is not at cap.
///
/// `Mutex<VecDeque>` is a small synchronous critical section — the pool
/// lock is held only for `pop_front` / `push_back` / `len`; we never
/// `.await` while holding it, so a `std::sync::Mutex` is safe.
type ChannelPool = Arc<Mutex<VecDeque<(mpsc::Sender<TokenEvent>, mpsc::Receiver<TokenEvent>)>>>;

/// Receiver wrapper that returns its (Sender, Receiver)
/// pair to [`EngineHandle::channel_pool`] on drop, instead of letting the
/// channel internals be deallocated.
///
/// Public API mirrors the subset of `tokio::sync::mpsc::Receiver` the wire
/// layer uses (`recv`, `try_recv`).  The wrapper is `Send` because all
/// internal state is `Send`; it is NOT `Sync` because `recv` requires
/// `&mut self`.
///
/// Drop discipline:
/// 1. Drain remaining events via `try_recv` until empty / closed (so the
///    next user of this channel does not see stale tokens from the
///    previous job).
/// 2. If the pool is below its cap, push `(return_sender, inner)` back to
///    the pool.  The pool's retained `Sender` keeps the channel alive
///    across requests.
/// 3. If the pool is full (overflow path), drop both endpoints — the
///    channel internals are deallocated normally.
pub struct PooledReceiver {
    inner: Option<mpsc::Receiver<TokenEvent>>,
    /// The `Sender` half of this pair, returned to the pool alongside the
    /// Receiver on drop.  `None` only after drop has run.
    return_sender: Option<mpsc::Sender<TokenEvent>>,
    /// Handle to the pool we return to on drop.  `None` for an unpooled
    /// receiver (used only in tests that bypass the pool, e.g.
    /// `collect_chat_from_events`).
    pool: Option<ChannelPool>,
    /// Pool size cap (`inbox_capacity + 1`).  Returns above this are
    /// dropped to avoid unbounded pool growth on overflow allocations.
    pool_cap: usize,
    /// Per-job cancellation guard. Present for every
    /// channel returned by [`EngineHandle::submit`]; `None` for the
    /// unpooled receivers used by in-process test helpers
    /// (`collect_chat_from_events`) which never need cancellation.
    /// Dropping `PooledReceiver` drops the guard, which flips the per-
    /// job [`CancellationFlag`] so the worker exits the decode loop
    /// within [`CANCEL_POLL_INTERVAL`] of the disconnect.
    cancel_guard: Option<CancellationGuard>,
}

impl PooledReceiver {
    /// Wrap a raw `Receiver` in a pooled receiver.  When the pool is
    /// `None`, the wrapper behaves like the raw receiver (no recycling on
    /// drop) — used for testing and `collect_chat_from_events`.
    ///
    /// `cancel_guard` is `Some` for every production
    /// receiver (returned by `EngineHandle::submit`); it is `None` for
    /// the unpooled test/inline receivers.  On `Drop` the guard flips
    /// the per-job [`CancellationFlag`] so the worker stops producing
    /// tokens promptly on stream end or client disconnect.
    pub(crate) fn new(
        rx: mpsc::Receiver<TokenEvent>,
        return_sender: mpsc::Sender<TokenEvent>,
        pool: Option<ChannelPool>,
        pool_cap: usize,
        cancel_guard: Option<CancellationGuard>,
    ) -> Self {
        Self {
            inner: Some(rx),
            return_sender: Some(return_sender),
            pool,
            pool_cap,
            cancel_guard,
        }
    }

    /// Receive the next event, awaiting if necessary.  Returns `None`
    /// when the channel is closed (no more senders alive — should not
    /// happen for a pooled channel because the pool holds a sender, but
    /// may happen if all senders including the pool's were dropped during
    /// shutdown).
    pub async fn recv(&mut self) -> Option<TokenEvent> {
        match self.inner.as_mut() {
            Some(rx) => rx.recv().await,
            None => None,
        }
    }

    /// Non-blocking receive; returns immediately with the next event, an
    /// `Empty` marker, or a `Disconnected` marker.  Used by `Drop` to
    /// drain remaining events before recycling.
    pub fn try_recv(&mut self) -> Result<TokenEvent, mpsc::error::TryRecvError> {
        match self.inner.as_mut() {
            Some(rx) => rx.try_recv(),
            None => Err(mpsc::error::TryRecvError::Disconnected),
        }
    }
}

impl Drop for PooledReceiver {
    fn drop(&mut self) {
        // Drop the cancellation guard FIRST so the worker
        // sees the cancel flag flip before our drain loop runs.  Without
        // this, a worker that is currently blocked in
        // `send_event_polling_cancel` keeps trying to push events while
        // we're draining — harmless for correctness (try_recv eats them)
        // but wastes one round-trip of polling.  Dropping first means
        // the worker observes cancel and stops feeding events within
        // CANCEL_POLL_INTERVAL (5 ms) of receiver drop.
        drop(self.cancel_guard.take());

        // Take ownership of the inner receiver and the return sender.
        // The sender clones held by the worker may still be alive briefly
        // (they drop at `process_job` exit), but the channel only closes
        // when ALL senders are dropped, so keeping `return_sender` alive
        // until we put it back into the pool guarantees no race.
        let (Some(mut rx), Some(tx)) = (self.inner.take(), self.return_sender.take()) else {
            return;
        };

        // Drain any leftover events so the next user of this channel
        // starts with an empty buffer.  Bounded by the channel capacity
        // (POOL_CHANNEL_CAPACITY = 16) PLUS the small number of events
        // the worker may emit between cancellation and observation
        // (≤ 1 token per ≤ CANCEL_POLL_INTERVAL), so this is at most
        // ~17 try_recv iterations — O(1) amortized per submit.
        loop {
            match rx.try_recv() {
                Ok(_) => continue,
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    // All senders gone (e.g. shutdown).  Don't recycle a
                    // dead channel; let it be deallocated by the
                    // outer scope.
                    return;
                }
            }
        }

        // Return to pool unless we're over cap.
        if let Some(pool) = self.pool.as_ref() {
            let mut guard = match pool.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            if guard.len() < self.pool_cap {
                guard.push_back((tx, rx));
            }
            // else: drop (tx, rx) at end of scope; pool is at cap.
        }
        // else: unpooled receiver; drop both endpoints normally.
    }
}

/// Cheap clonable handle handlers use to submit jobs.
#[derive(Clone)]
pub struct EngineHandle {
    sender: mpsc::Sender<WorkerJob>,
    model_info: Arc<ModelInfo>,
    tokenizer: Arc<dyn Tokenize>,
    /// shared per-component memory breakdown. The worker writes
    /// after each job; the `/debug/memory_breakdown` endpoint reads via
    /// [`EngineHandle::memory_breakdown_snapshot`].  Defaults to the
    /// all-zero snapshot until the first post-job update.
    breakdown: Arc<Mutex<ServerMemoryBreakdown>>,
    /// Pre-allocated pool of `(Sender, Receiver)` pairs,
    /// recycled across requests to eliminate per-request channel
    /// allocations.  See [`PooledReceiver`] and the module-level
    /// `POOL_CHANNEL_CAPACITY` constant.
    channel_pool: ChannelPool,
    /// Pool size cap = `inbox_capacity + 1`.  Sits next to `channel_pool`
    /// so [`PooledReceiver::drop`] can read it without an extra lock.
    pool_cap: usize,
}

impl EngineHandle {
    /// Submit a job to the worker. Returns the receiver of token events.
    /// The receiver closes when the worker is done with the job and the
    /// pool's retained `Sender` clone is dropped (only at shutdown).
    ///
    /// `event_buffer` is**advisory only**. The channel-pool path recycles
    /// pre-allocated channels of fixed capacity `POOL_CHANNEL_CAPACITY`
    /// (16) drawn from a per-worker pool.  When `event_buffer` exceeds
    /// the pool's channel capacity (e.g. legacy callers pass 128), the
    /// excess capacity is unused but no error is returned — the
    /// signature is preserved for backward compatibility with callers
    /// that may pass smaller values for less-buffer scenarios.
    ///
    /// On pool exhaustion (all pre-allocated pairs in flight), the
    /// allocator falls back to a fresh `mpsc::channel(event_buffer)`,
    /// which is dropped on receiver completion rather than returned to
    /// the pool (overflow paths don't grow the pool indefinitely).
    pub async fn submit(
        &self,
        request: JobRequest,
        event_buffer: usize,
    ) -> Result<JobResponseChannel, ServerError> {
        let (tx, rx, return_sender, pool_handle) = self.take_channel_pair(event_buffer);
        // Stage the channel pair in a `PooledReceiver` first.  If the
        // worker send fails below, we drop the receiver and the channel
        // pair is NOT recycled (because the worker is gone — we don't
        // want zombie pairs in the pool).  We achieve that by giving the
        // PooledReceiver `pool=None` in the error path.
        let pool_cap = self.pool_cap;

        // Allocate per-job cancellation flag. The worker
        // gets a clone via `WorkerJob.cancel`; the wire layer gets the
        // companion `CancellationGuard` baked into the returned
        // `PooledReceiver` so dropping the receiver (handler return,
        // client disconnect, panic) flips the flag.  See module-level
        // doc on `CANCEL_POLL_INTERVAL` for the polling cadence and on
        // `CancellationFlag` for the memory-ordering rationale.
        //
        // This allocation is on the hot path
        // of every submit; one `Arc::new(AtomicBool::new(false))` per
        // request is a single small heap alloc + one clone.  The
        // `Arc` pair drops cleanly when both the worker (process_job
        // exit) and the wire layer (`PooledReceiver::drop`) have
        // released their clones.  If the supervisor's `catch_unwind`
        // arm fires before the worker clone drops, the worker's clone
        // is unwound as part of the panic stack — the wire-side guard
        // then drops normally when the body task observes the error,
        // setting the flag to `true` for any future borrower (none, in
        // practice; the channel pair is recycled afresh by the next
        // `submit`).
        let cancel: CancellationFlag = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let guard = CancellationGuard::new(cancel);

        match self
            .sender
            .send(WorkerJob {
                request,
                tokens_tx: tx,
                cancel: worker_cancel,
            })
            .await
        {
            Ok(()) => Ok(PooledReceiver::new(
                rx,
                return_sender,
                pool_handle,
                pool_cap,
                Some(guard),
            )),
            Err(_) => {
                // Worker is gone.  Drop everything; do NOT return to pool
                // even if `pool_handle.is_some()`, because the channel
                // pair's `tx.clone()` we just gave the worker is dead-on-
                // arrival and the pair is no longer in a sane state.
                // Drop the guard too — there is no live worker to observe
                // the cancel flag flip; the AtomicBool just deallocates.
                drop(guard);
                drop(return_sender);
                drop(rx);
                drop(pool_handle);
                Err(ServerError::EngineUnavailable("worker channel closed".into()))
            }
        }
    }

    /// Take a `(Sender, Receiver)` pair from the pool,
    /// or allocate a fresh one if the pool is empty.
    ///
    /// Returns:
    /// - `tx`: a `Sender` clone for the worker to publish events.
    /// - `rx`: the `Receiver` for the wire layer to consume events.
    /// - `return_sender`: the *original* Sender retained by the pool,
    ///   passed to [`PooledReceiver`] so it can be returned to the pool
    ///   on drop.  Distinct from `tx` (which is a clone) — keeping
    ///   both alive across the request guarantees the channel never
    ///   closes between worker-Sender-drop and PooledReceiver-drop.
    /// - `pool_handle`: `Some(pool)` when the pair came from the pool
    ///   (will be recycled on drop), `None` for overflow allocations
    ///   (dropped at scope exit, not recycled).
    fn take_channel_pair(
        &self,
        event_buffer: usize,
    ) -> (
        mpsc::Sender<TokenEvent>,
        mpsc::Receiver<TokenEvent>,
        mpsc::Sender<TokenEvent>,
        Option<ChannelPool>,
    ) {
        // Fast path: try to pop from pool.  Held briefly; never `.await`s.
        let popped = {
            let mut guard = match self.channel_pool.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            guard.pop_front()
        };
        match popped {
            Some((return_sender, rx)) => {
                // Pool hit: clone the sender for the worker, return the
                // original (which will go back into the pool) plus rx.
                let worker_tx = return_sender.clone();
                (worker_tx, rx, return_sender, Some(Arc::clone(&self.channel_pool)))
            }
            None => {
                // Pool miss (overflow): allocate a fresh pair.  Honors
                // the `event_buffer` arg here so legacy 128 still works.
                // The fresh pair is NOT returned to the pool on drop —
                // overflow allocations are throwaway by design.
                let cap = event_buffer.max(1);
                let (tx, rx) = mpsc::channel(cap);
                let worker_tx = tx.clone();
                (worker_tx, rx, tx, None)
            }
        }
    }

    /// Snapshot of the channel pool's current depth.
    /// Used by `update_memory_breakdown` and tests; never blocks more
    /// than briefly.
    pub fn channel_pool_len(&self) -> usize {
        match self.channel_pool.lock() {
            Ok(g) => g.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    /// Get the static info for the model this worker is serving.
    pub fn model_info(&self) -> Arc<ModelInfo> {
        Arc::clone(&self.model_info)
    }

    /// Maximum context length (in tokens) the loaded model / KV cache admits.
    ///
    /// Backed by `model_info.context_length`, which is set from the server's
    /// `context_length` at startup and equals the worker's
    /// `config.max_seq_len` (both flow from the same value in
    /// `lumen-server.rs`). Exposed so the wire layer can reject an over-long
    /// prompt SYNCHRONOUSLY in `into_job` — returning a 400 before the 200 OK
    /// / SSE stream opens — instead of relying solely on the worker's
    /// backstop guard (which, on the streaming path, can only surface as a
    /// mid-stream error frame after headers are already sent).
    pub fn context_length(&self) -> usize {
        self.model_info.context_length
    }

    /// Test-only: build a minimal `EngineHandle` backed by an
    /// [`IdentityByteTokenizer`] and NO live worker, so wire-layer unit tests
    /// can exercise `into_job` / `context_length` / `tokenize_for_request`
    /// without booting an engine. `submit` is unusable on this handle (the
    /// worker sender is dropped), which is fine — `into_job` never calls it.
    /// `context_length` is configurable so the synchronous oversize guard can
    /// be unit-tested at a chosen window.
    #[cfg(test)]
    pub(crate) fn new_for_test(context_length: usize) -> Self {
        // Capacity-1 channel whose receiver is dropped immediately: no worker
        // consumes jobs, but `into_job` only reads tokenizer / model_info.
        let (sender, _job_inbox_rx) = mpsc::channel::<WorkerJob>(1);
        Self {
            sender,
            model_info: Arc::new(ModelInfo {
                id: "lumen-test:unit".into(),
                owned_by: "lumen-test".into(),
                created: 0,
                context_length,
            }),
            tokenizer: Arc::new(IdentityByteTokenizer::default()) as Arc<dyn Tokenize>,
            breakdown: Arc::new(Mutex::new(ServerMemoryBreakdown::default())),
            channel_pool: Arc::new(Mutex::new(VecDeque::new())),
            pool_cap: 2,
        }
    }

    /// Tokenize a rendered prompt string. Called by request handlers.
    pub fn tokenize_for_request(&self, prompt: &str) -> Vec<u32> {
        self.tokenizer.encode(prompt)
    }

    /// EOS token ids for the loaded model.
    pub fn eos_tokens_for_request(&self) -> Vec<u32> {
        self.tokenizer.eos_tokens()
    }

    /// Apply the loaded model's chat template, if the tokenizer overrides
    /// the default. Returns `None` if the tokenizer asks the caller to do
    /// formatting itself.
    pub fn apply_chat_template(&self, system: Option<&str>, user: &str) -> Option<String> {
        self.tokenizer.apply_chat_template(system, user)
    }

    /// Read the latest per-component memory breakdown captured
    /// after the most recent completed job.
    ///
    /// Returns a `Clone` of the breakdown so handlers don't hold the
    /// internal mutex across `.await` boundaries.  The snapshot is
    /// best-effort lag-bounded: if no job has completed since the worker
    /// last updated, `update_count` and `last_update_unix` reflect that
    /// stale state.  All-zeros mean the worker has not completed its first
    /// job yet.
    ///
    /// Cost: one `lock().clone()`.  Mutex contention is between the worker
    /// thread (writes once per job) and HTTP handlers (read once per
    /// sample); poisoning is impossible in practice because both sides hold
    /// the lock briefly and don't panic inside the critical section, but
    /// we handle poisoning by returning the inner state — the breakdown is
    /// non-critical telemetry, not a correctness invariant.
    pub fn memory_breakdown_snapshot(&self) -> ServerMemoryBreakdown {
        match self.breakdown.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// internal — clone the breakdown Arc for handler wiring.
    /// Used by `build_router` to expose the breakdown via the HTTP
    /// endpoint without going through `submit`.
    #[doc(hidden)]
    pub fn breakdown_arc(&self) -> Arc<Mutex<ServerMemoryBreakdown>> {
        Arc::clone(&self.breakdown)
    }
}

/// Static metadata about the loaded model. Exposed via `/v1/models`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    /// User-facing model id (e.g. `"qwen3.5-9b:q8_0"`).
    pub id: String,
    /// Best-effort owner label for the OpenAI compatibility surface.
    pub owned_by: String,
    /// Unix timestamp the model was loaded.
    pub created: u64,
    /// Maximum context length supported by the loaded weights.
    pub context_length: usize,
}

/// Trait the worker uses to decode tokens back to text for streaming.
///
/// The server is tokenizer-agnostic. Callers supply an implementation that
/// wraps whatever BPE / SPM library they use.
pub trait Tokenize: Send + Sync + 'static {
    /// Encode `text` into token ids. Returns the ids ready to feed into the
    /// session (chat-template handling is the caller's responsibility).
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Incrementally decode a single token, given the byte-state held over
    /// from prior tokens. Returns the new fragment of decoded text and
    /// the updated byte state.
    ///
    /// The state is opaque bytes the implementor uses to buffer partial
    /// UTF-8 / partial BPE-byte sequences across token boundaries. A
    /// reasonable convention is to keep raw bytes here and re-decode on
    /// each call.
    fn decode_incremental(&self, state: &mut Vec<u8>, token_id: u32) -> String;

    /// Decode a SINGLE token id to its raw bytes (no UTF-8 buffering / no
    /// lossy substitution). Powers the greedy anti-restate guard's sub-word
    /// doubling rule, which needs the bytes of the candidate and of the
    /// just-emitted token to recognise " multiplication" → "lication".
    ///
    /// Default impl returns an empty `Vec` so tokenizers that do not override
    /// it simply disable the byte-level rule (the id-only n-gram restate rule
    /// still works). Real BPE/SPM tokenizers should return the token's
    /// per-id bytes (e.g. `BpeTokenizer::decode_bytes(&[id])`).
    fn decode_id_bytes(&self, _token_id: u32) -> Vec<u8> {
        Vec::new()
    }

    /// Encode + apply chat template for a system+user pair. Default impl
    /// returns `None` so callers that just want raw completion don't have
    /// to override.
    fn apply_chat_template(&self, _system: Option<&str>, _user: &str) -> Option<String> {
        None
    }

    /// EOS / stop-token ids for this tokenizer. The worker stops decoding
    /// when one of these is sampled.
    fn eos_tokens(&self) -> Vec<u32>;
}

/// The worker that owns the runtime engine. Spawn one per server instance
/// via [`EngineWorker::spawn`]; the returned [`EngineHandle`] is what HTTP
/// handlers use.
pub struct EngineWorker {
    config: RuntimeConfig,
    hyperparams: ModelHyperparams,
    backend: Box<dyn ComputeBackend>,
    weights: Arc<dyn WeightProvider>,
    tokenizer: Arc<dyn Tokenize>,
    /// Retained on the worker for log/telemetry hooks; the public-facing
    /// copy lives on [`EngineHandle`].
    #[allow(dead_code)]
    model_info: Arc<ModelInfo>,
    inbox: mpsc::Receiver<WorkerJob>,
    /// Inbox capacity captured at spawn time so the post-job breakdown
    /// update can record it without re-querying the mpsc channel.
    inbox_capacity: usize,
    /// Optional disk-persistent KV cache configuration.
    disk_kv: Option<DiskKvConfig>,
    /// per-component memory breakdown shared with [`EngineHandle`].
    /// The worker writes after every completed job; HTTP handlers read via
    /// [`EngineHandle::memory_breakdown_snapshot`].
    breakdown: Arc<Mutex<ServerMemoryBreakdown>>,
}

impl EngineWorker {
    /// Create a worker and start its background task. Returns the handle
    /// callers should hand to [`crate::router::build_router`].
    ///
    /// Backpressure: `inbox_size` is the bounded capacity of the job inbox;
    /// once that many jobs are queued, `EngineHandle::submit` will block
    /// (in async tokio sense, yield) until the worker drains. 16 is a
    /// reasonable default.
    pub fn spawn(
        config: RuntimeConfig,
        hyperparams: ModelHyperparams,
        backend: Box<dyn ComputeBackend>,
        weights: Arc<dyn WeightProvider>,
        tokenizer: Arc<dyn Tokenize>,
        model_info: ModelInfo,
        inbox_size: usize,
    ) -> EngineHandle {
        Self::spawn_with_disk_cache(
            config, hyperparams, backend, weights, tokenizer, model_info, inbox_size, None,
        )
    }

    /// Same as [`Self::spawn`] but lets the caller wire a disk-persistent
    /// KV cache. When `disk_kv` is `Some`, the worker performs
    /// the startup housekeeping documented on [`DiskKvConfig`].
    pub fn spawn_with_disk_cache(
        config: RuntimeConfig,
        hyperparams: ModelHyperparams,
        backend: Box<dyn ComputeBackend>,
        weights: Arc<dyn WeightProvider>,
        tokenizer: Arc<dyn Tokenize>,
        model_info: ModelInfo,
        inbox_size: usize,
        disk_kv: Option<DiskKvConfig>,
    ) -> EngineHandle {
        let capacity = inbox_size.max(1);
        let (tx, rx) = mpsc::channel::<WorkerJob>(capacity);
        let model_info = Arc::new(model_info);
        // initialise breakdown with engine_inbox_capacity already
        // populated so even a snapshot taken before the first completed
        // job carries the constant fields.  All other fields stay zero
        // until process_job has run at least once.
        let breakdown = Arc::new(Mutex::new(ServerMemoryBreakdown {
            engine_inbox_capacity: capacity,
            ..ServerMemoryBreakdown::default()
        }));
        // Pre-allocate the channel pool with `capacity`
        // pairs of `POOL_CHANNEL_CAPACITY` slots each.  Cap the pool at
        // `capacity + 1` so the steady-state pool depth fits within the
        // worker's "one-job-in-flight + capacity-1 queued" invariant
        // without an unbounded grow.
        let pool_cap = capacity.saturating_add(1);
        let mut pool_deque = VecDeque::with_capacity(pool_cap);
        for _ in 0..capacity {
            let pair = mpsc::channel::<TokenEvent>(POOL_CHANNEL_CAPACITY);
            pool_deque.push_back(pair);
        }
        let channel_pool: ChannelPool = Arc::new(Mutex::new(pool_deque));
        let worker = Self {
            config,
            hyperparams,
            backend,
            weights,
            tokenizer: Arc::clone(&tokenizer),
            model_info: Arc::clone(&model_info),
            inbox: rx,
            inbox_capacity: capacity,
            disk_kv,
            breakdown: Arc::clone(&breakdown),
        };
        tokio::task::spawn_blocking(move || worker.run());
        EngineHandle {
            sender: tx,
            model_info,
            tokenizer,
            breakdown,
            channel_pool,
            pool_cap,
        }
    }

    /// Main loop. Runs on a `spawn_blocking` thread because the inner
    /// `Session::next_token` call can take milliseconds and must not block
    /// the tokio executor.
    fn run(mut self) {
        // The session is per-worker, NOT per-job, so suffix prefill can
        // detect shared prefixes across consecutive jobs. (For multi-tenant
        // servers, swap to a session pool keyed by conversation id; the
        // single-session design is correct for the single-client agent use
        // case Lumen targets first.)
        let session = Session::new(
            self.config.clone(),
            self.hyperparams,
            SamplingParams::default(),
        );
        let mut session = match session {
            Ok(s) => s,
            Err(e) => {
                // Drain inbox with errors and shut down.
                while let Some(job) = self.inbox.blocking_recv() {
                    let _ = job
                        .tokens_tx
                        .blocking_send(TokenEvent::Error(format!("session init failed: {e}")));
                }
                return;
            }
        };

        // Install the per-token-id byte decoder for the greedy anti-restate
        // guard's sub-word-doubling rule. The closure captures an `Arc` clone
        // of the worker's tokenizer and maps a token id to its raw decoded
        // bytes. When the guard is disabled (dense, or LUMEN_ANTI_RESTATE=0)
        // the closure is simply never invoked.
        {
            let tok = Arc::clone(&self.tokenizer);
            session.set_token_decoder(Arc::new(move |id: u32| tok.decode_id_bytes(id)));
        }

        // validate KV precision against backend once at worker
        // startup so a misconfigured server fails fast with a clear error
        // (Metal requires F16, CUDA requires F32) instead of silently
        // corrupting KV writes inside the first job.
        if let Err(e) = session.validate_backend(self.backend.as_ref()) {
            while let Some(job) = self.inbox.blocking_recv() {
                let _ = job
                    .tokens_tx
                    .blocking_send(TokenEvent::Error(format!(
                        "backend / KV precision mismatch: {e}"
                    )));
            }
            return;
        }

        // Disk-KV startup housekeeping.
        // --kv-disk-dir without --kv-disk-space-mb runs in purge-only mode:
        // .tmp.<pid> files are cleaned but .kv entries grow unbounded.
        // Surface this once at startup so the operator can spot it.
        if let Some(cfg) = self.disk_kv.as_ref() {
            if cfg.budget_bytes == 0 || cfg.budget_bytes == u64::MAX {
                eprintln!(
                    "[server kv-disk] {} is configured without --kv-disk-space-mb: \
                     .kv files will accumulate unbounded; only .tmp.<pid> orphans are purged. \
                     Set --kv-disk-space-mb to enable eviction.",
                    cfg.dir.display(),
                );
            }
        }
        // startup has no live prefix yet (session is empty),
        // so the pinning set is also empty.
        self.disk_kv_housekeeping("startup", &session);

        // cache whether the per-component breakdown is enabled
        // for THIS worker lifetime.  When unset/0 the post-job update is
        // skipped entirely so the default path performs zero additional
        // work and the default response bytes are unchanged. Operators
        // who flip the env var mid-run will need to restart the worker;
        // this is a deliberate trade for a single env lookup per worker
        // rather than one per job.
        let breakdown_enabled = match std::env::var("LUMEN_SERVER_DEBUG_MEM") {
            Ok(v) => !v.is_empty() && v != "0",
            Err(_) => false,
        };
        // panic supervisor. A rolling deque of recent
        // panic timestamps caps the recovery budget at
        // `max_panics_in_window()` events inside `panic_window()`.
        // The thresholds are resolved once per worker so an operator
        // edit during a long-running process never mid-flight changes
        // the budget; restart to apply.
        let panic_window_dur = panic_window();
        let panic_budget = max_panics_in_window();
        let mut recent_panics: VecDeque<Instant> = VecDeque::with_capacity(panic_budget + 1);
        while let Some(job) = self.inbox.blocking_recv() {
            // Snapshot the reply sender BEFORE moving `job` into
            // `process_job` so the panic arm can publish a clean
            // 503-shaped `Error` event to the in-flight client.  The
            // `mpsc::Sender` is cheaply clonable; the underlying
            // channel pair is owned by the [`PooledReceiver`] on the
            // wire side and stays alive until that drop.
            let reply_tx = job.tokens_tx.clone();
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.process_job(&mut session, job);
            }));
            if let Err(payload) = result {
                // Per-job panic.  Notify the in-flight client first so
                // it doesn't hang on `recv().await`, then decide
                // whether to recover or shut down.
                let msg = panic_payload_message(payload.as_ref());
                let _ = reply_tx.blocking_send(TokenEvent::Error(format!(
                    "engine recovered from panic: {msg}"
                )));
                drop(reply_tx);
                let now = Instant::now();
                // Drop timestamps older than the rolling window.
                while let Some(&ts) = recent_panics.front() {
                    if now.duration_since(ts) > panic_window_dur {
                        recent_panics.pop_front();
                    } else {
                        break;
                    }
                }
                recent_panics.push_back(now);
                eprintln!(
                    "[server engine] worker recovered from panic ({} in last {}s): {msg}",
                    recent_panics.len(),
                    panic_window_dur.as_secs(),
                );
                if recent_panics.len() > panic_budget {
                    eprintln!(
                        "[server engine] panic budget exhausted ({} > {} in {}s); marking \
                         engine UNHEALTHY and draining inbox",
                        recent_panics.len(),
                        panic_budget,
                        panic_window_dur.as_secs(),
                    );
                    while let Some(stale) = self.inbox.blocking_recv() {
                        let _ = stale.tokens_tx.blocking_send(TokenEvent::Error(
                            "engine unhealthy: too many panics; restart required".into(),
                        ));
                    }
                    return;
                }
                // Rebuild the per-worker `Session` because its KV
                // scratch may have been left mid-write by the failed
                // kernel.  Weights are shared `Arc`s — only KV
                // buffers + sampler state are reallocated; the cost
                // is bounded and dominated by `Vec::with_capacity` on
                // recovery, not by re-staging weights.
                match Session::new(
                    self.config.clone(),
                    self.hyperparams,
                    SamplingParams::default(),
                ) {
                    Ok(s) => {
                        session = s;
                        if let Err(e) = session.validate_backend(self.backend.as_ref()) {
                            eprintln!(
                                "[server engine] re-validate_backend after panic failed: {e}; \
                                 draining inbox"
                            );
                            while let Some(stale) = self.inbox.blocking_recv() {
                                let _ = stale.tokens_tx.blocking_send(TokenEvent::Error(format!(
                                    "engine unhealthy: backend revalidation failed: {e}"
                                )));
                            }
                            return;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[server engine] Session::new after panic failed: {e}; \
                             draining inbox"
                        );
                        while let Some(stale) = self.inbox.blocking_recv() {
                            let _ = stale.tokens_tx.blocking_send(TokenEvent::Error(format!(
                                "engine unhealthy: session reinit failed: {e}"
                            )));
                        }
                        return;
                    }
                }
                // Skip the post-job housekeeping for the panicking
                // job — session was rebuilt; nothing useful to
                // breakdown / evict against the dead state.
                continue;
            }
            // After every job, run a cheap eviction sweep so a long-running
            // server with a tight budget never grows past it. The sweep
            // returns immediately when the directory is already under
            // budget. The session's current token prefix is passed in so
            // its on-disk twin (if any) gets the live-session 0.25× penalty
            // rather than being evicted out from under us.
            self.disk_kv_housekeeping("post-job", &session);
            // refresh the per-component memory breakdown so the
            // /debug/memory_breakdown HTTP endpoint reads fresh data on
            // its next sample.  Runs on the worker thread (already off
            // the tokio executor) so the slower disk-walk and Metal
            // driver query never block request handlers.  Skipped when
            // the endpoint is disabled so the default path stays
            // bit-exact when disabled.
            if breakdown_enabled {
                self.update_memory_breakdown(&session);
            }
        }
    }

    /// refresh the per-component memory breakdown.
    ///
    /// Called once after each completed job.  Captures the LIVE session
    /// state (KV bytes, token-history length, pending-logits / per-layer
    /// timing buffers), the backend's driver-side allocated bytes
    /// (`MTLDevice.currentAllocatedSize` on Metal), the Tokio runtime's
    /// `num_alive_tasks()` metric when available, the engine inbox depth,
    /// and the disk-KV directory size when a disk cache is configured.
    ///
    /// Cost: one Mutex acquisition, one Metal driver query, one statvfs
    /// directory walk (skipped when disk_kv is None or budget == 0).  All
    /// off the request-handler hot path; runs on the worker thread between
    /// jobs.
    ///
    /// This method writes a complete snapshot every call (no per-field
    /// deltas) so a poisoned mutex or stale field cannot accumulate over
    /// time.
    fn update_memory_breakdown(&self, session: &Session) {
        let kv = session.kv();
        let kv_used_bytes = kv.total_bytes();
        let kv_allocated_bytes = kv.allocated_bytes();
        let kv_seq_len = kv.seq_len();
        let kv_max_seq_len = kv.max_seq_len();

        let session_tokens_len = session.token_count();
        // The hyperparams snapshot has the vocab dimension we need to
        // estimate `pending_logits` cost without exposing the field on
        // `Session`.  `Session::tokens()` is a borrow; we don't need the
        // logits themselves, only the footprint, which is `vocab_size *
        // 4` bytes when the logits buffer is held between calls (the
        // common case for the suffix-prefill path after the first job).
        let vocab_size = self.hyperparams.vocab_size as u64;
        // Logits is `Option<Logits>`; we cannot inspect it here without
        // adding a Session accessor.  Approximate as `vocab_size * 4`
        // bytes per-job because that is what `Session::extend` stashes
        // into `pending_logits`.  After the next-token decode loop empties
        // it, the buffer is None again; capture the steady-state worst
        // case so the sampler does not over- or under-report the leak.
        let session_pending_logits_bytes = vocab_size * 4;

        // `Session::take_timings` would drain the buffer; we only want a
        // size estimate.  Per-layer timings collection is off by default
        // (`RuntimeConfig.collect_per_layer_timings = false`), so this is
        // 0 in the soak harness — but if a future operator turns it on
        // the breakdown surfaces the growth.  Worst-case size is
        // `num_layers * generated_tokens * size_of::<PerLayerTiming>()`,
        // and we already track generated_tokens via session.token_count.
        // For now we report a conservative 0 unless `collect_per_layer_timings`
        // is on, in which case we estimate from token_count.
        let session_timings_bytes = if self.config.collect_per_layer_timings {
            (session_tokens_len as u64)
                * (self.hyperparams.num_layers as u64)
                * (std::mem::size_of::<lumen_runtime::PerLayerTiming>() as u64)
        } else {
            0
        };

        let metal_current_allocated_bytes = self.backend.current_allocated_bytes();

        let tokio_active_tasks = current_tokio_alive_tasks();

        // Inbox depth: `Receiver::len()` is the count of un-popped items.
        let engine_inbox_len = self.inbox.len();
        let engine_inbox_capacity = self.inbox_capacity;

        // Disk-KV bytes via `dir_size_recursive` ONLY when configured.  A
        // quiet-or-absent disk-KV produces 0 (no I/O wasted).
        let disk_kv_used_bytes = match self.disk_kv.as_ref() {
            Some(cfg) => dir_size_recursive(&cfg.dir),
            None => 0,
        };

        let now_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut guard = match self.breakdown.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.kv_used_bytes = kv_used_bytes;
        guard.kv_allocated_bytes = kv_allocated_bytes;
        guard.kv_seq_len = kv_seq_len;
        guard.kv_max_seq_len = kv_max_seq_len;
        guard.session_tokens_len = session_tokens_len;
        guard.session_pending_logits_bytes = session_pending_logits_bytes;
        guard.session_timings_bytes = session_timings_bytes;
        guard.metal_current_allocated_bytes = metal_current_allocated_bytes;
        guard.tokio_active_tasks = tokio_active_tasks;
        guard.engine_inbox_capacity = engine_inbox_capacity;
        guard.engine_inbox_len = engine_inbox_len;
        guard.disk_kv_used_bytes = disk_kv_used_bytes;
        guard.update_count = guard.update_count.saturating_add(1);
        guard.last_update_unix = now_unix;
    }

    /// Run the configured disk-KV housekeeping pass: purge stale `.tmp.<pid>`
    /// files left behind by killed `save_atomic` writers and then evict
    /// lowest-scoring entries until the directory fits the budget.
    ///
    /// `phase` is a short label for the log line so the operator can
    /// distinguish the startup sweep from per-job sweeps.
    ///
    /// builds the `live_session_prefixes` set from the
    /// session's current tokens. This protects the in-memory session's
    /// on-disk twin from being evicted while the live session is still
    /// extending it (the eviction policy gives such entries a 0.25×
    /// score penalty instead of skipping them entirely; if the cache is
    /// genuinely starving for budget the live entry can still go, but
    /// only after every cold entry has been evicted).
    fn disk_kv_housekeeping(&self, phase: &str, session: &Session) {
        let Some(cfg) = self.disk_kv.as_ref() else { return };
        match lumen_runtime::kv::disk::purge_stale_tmp(&cfg.dir) {
            Ok(n) if n > 0 => {
                eprintln!(
                    "[server kv-disk/{phase}] purged {n} stale .tmp.<pid> file(s) from {}",
                    cfg.dir.display(),
                );
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "[server kv-disk/{phase}] purge_stale_tmp({}) failed: {e}",
                    cfg.dir.display(),
                );
            }
        }
        // A budget of 0 / u64::MAX means "no eviction, only orphan purging".
        if cfg.budget_bytes > 0 && cfg.budget_bytes < u64::MAX {
            // M4: build the live-prefix pin set from the session's
            // current tokens. Empty-token sessions (startup) skip the pin.
            let mut live_prefixes = std::collections::HashSet::new();
            let tokens = session.tokens();
            if !tokens.is_empty() {
                let name = lumen_runtime::kv::disk::cache_filename(tokens);
                if let Some(stem) = name.strip_suffix(".kv") {
                    live_prefixes.insert(stem.to_string());
                }
            }
            match lumen_runtime::kv::disk::evict_to_budget(
                &cfg.dir,
                cfg.budget_bytes,
                &live_prefixes,
            ) {
                Ok(removed) if !removed.is_empty() => {
                    eprintln!(
                        "[server kv-disk/{phase}] eviction removed {} file(s) to fit budget {} bytes",
                        removed.len(),
                        cfg.budget_bytes,
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        "[server kv-disk/{phase}] evict_to_budget({}) failed: {e}",
                        cfg.dir.display(),
                    );
                }
            }
        }
    }

    /// Cancel-aware token send.
    ///
    /// Replaces `tokens_tx.blocking_send(event).is_err()` everywhere the
    /// worker emits an event during the decode hot path.  The standard
    /// `blocking_send` blocks indefinitely when the channel is full as
    /// long as ANY `Sender` clone is alive — and the channel pool
    /// retains a `return_sender` clone for the channel's lifetime, so
    /// `blocking_send` can never observe an "all senders dropped" close
    /// signal.  A client that disconnects mid-stream would wedge the
    /// single-worker engine: 16 tokens fill the channel, then
    /// `blocking_send` parks forever and no subsequent request can
    /// be processed.
    ///
    /// To bound that, this method:
    /// 1. Tries a non-blocking `try_send` first.  In the common case
    ///    (channel has slack, receiver is healthy) this returns `Ok(())`
    ///    immediately.
    /// 2. On `Full`, polls the per-job cancellation flag every
    ///    [`CANCEL_POLL_INTERVAL`] (5 ms).  When the flag flips —
    ///    [`CancellationGuard::drop`] flips it on stream end or client
    ///    disconnect — the method returns `Err(())` so the caller breaks
    ///    out of the decode loop and clean-exits the job.
    /// 3. On `Closed` (worker shutdown or unpooled-test channel close),
    ///    also returns `Err(())`.
    ///
    /// Returns `Ok(())` iff the event was successfully delivered;
    /// `Err(())` iff the worker should abandon further sends for this
    /// job (cancellation or channel closed).
    ///
    /// Interaction with the panic supervisor: a panic during the
    /// channel-full poll would be caught by the outer `catch_unwind` in
    /// `EngineWorker::run`; the cancel flag is untouched, but the
    /// wire-side guard drops normally when the 503 `Error` event is
    /// observed by the body task.  The flag flip is therefore observable
    /// by any lingering sender clones
    /// once the channel pair is recycled; in practice the recycle
    /// path resets the flag implicitly because the *next* `submit`
    /// allocates a fresh `Arc<AtomicBool>`.
    fn send_event_polling_cancel(
        &self,
        tokens_tx: &mpsc::Sender<TokenEvent>,
        cancel: &AtomicBool,
        mut event: TokenEvent,
    ) -> Result<(), ()> {
        // Cheap fast-path cancel check before the first send attempt.
        // If the wire layer already dropped the guard (e.g. very fast
        // client disconnect during prefill), skip the channel altogether.
        if cancel.load(Ordering::Relaxed) {
            return Err(());
        }
        loop {
            match tokens_tx.try_send(event) {
                Ok(()) => return Ok(()),
                Err(mpsc::error::TrySendError::Full(returned)) => {
                    // Channel full.  Re-take ownership of the event,
                    // poll for cancel, sleep briefly, retry.
                    event = returned;
                    if cancel.load(Ordering::Relaxed) {
                        return Err(());
                    }
                    std::thread::sleep(CANCEL_POLL_INTERVAL);
                }
                Err(mpsc::error::TrySendError::Closed(_)) => return Err(()),
            }
        }
    }

    fn process_job(&self, session: &mut Session, job: WorkerJob) {
        let WorkerJob {
            request,
            tokens_tx,
            cancel,
        } = job;
        // Move the per-request sampling params (temperature / seed) onto the
        // long-lived per-worker session for this job and re-seed its RNG.
        // The session is constructed once in `run()` with
        // `SamplingParams::default()` (temperature=1.0, advancing RNG); without
        // this apply, every request would sample at the default temperature
        // with a never-reset RNG → incoherent + non-deterministic output.
        // Re-seeding here means two byte-identical
        // `temperature:0, seed:N` requests decode identically.
        session.set_sampling(request.sampling.clone());
        let prior_tokens = session.token_count();

        // Prompt-length guard: reject prompts longer than the KV's max_seq_len
        // BEFORE they reach the prefill kernel. Without this guard, the GPU
        // kernel would write to KV positions beyond the allocated buffer end —
        // undefined behaviour on Metal and out-of-bounds on CUDA. The CLI side
        // pre-sizes max_seq_len via `effective_max_seq_len` in `lumen-cli`, but
        // the server's max_seq_len is admin-controlled at startup while
        // prompts are client-controlled per request, so this check is the
        // last line of defence.
        if request.prompt_tokens.len() > self.config.max_seq_len {
            let _ = self.send_event_polling_cancel(
                &tokens_tx,
                &cancel,
                TokenEvent::Error(format!(
                    "prompt is {} tokens but server max_seq_len is {}; reduce prompt or restart with a larger --context-len",
                    request.prompt_tokens.len(),
                    self.config.max_seq_len,
                )),
            );
            return;
        }

        // Suffix prefill: reuses KV when the new prompt extends the prior
        // job's tokens; falls back to cold prefill otherwise.
        let suffix_result: Result<SuffixPrefillResult, RuntimeError> = session
            .extend_with_cache(
                &request.prompt_tokens,
                self.backend.as_ref(),
                self.weights.as_ref(),
                request.suffix_threshold.max(1),
            );

        let prefill = match suffix_result {
            Ok(r) => r,
            Err(e) => {
                let _ = self.send_event_polling_cancel(
                    &tokens_tx,
                    &cancel,
                    TokenEvent::Error(format!("prefill failed: {e}")),
                );
                return;
            }
        };

        if self
            .send_event_polling_cancel(
                &tokens_tx,
                &cancel,
                TokenEvent::PrefillDone {
                    reused_prefix_len: prefill.reused_prefix_len,
                    suffix_len: prefill.suffix_len,
                    prefill_time: prefill.prefill_time,
                },
            )
            .is_err()
        {
            return; // client gave up or worker shutting down
        }

        // Decode loop.
        let mut bytes_state: Vec<u8> = Vec::new();
        let mut generated = 0usize;
        // -- F4: textual stop-sequence matcher over the rolling DECODED text.
        //
        // Seeded from `request.stop_text` (the OpenAI `stop` / Anthropic
        // `stop_sequences` list). When a stop string appears in the cumulative
        // answer text, the loop emits the prefix UP-TO-BUT-EXCLUDING the stop
        // and terminates with `FinishReason::StopSequence`. The matcher buffers the
        // ambiguous tail across token boundaries so a stop straddling two
        // decoded fragments is still caught (same window logic the wire layer
        // uses, so worker and wire agree byte-for-byte).
        //
        // EMPTY-STOP BYTE-IDENTITY: when `request.stop_text` is empty,
        // `StopMatcher::push` returns `(fragment, false)` verbatim and
        // `finish()` returns "" — so every emitted `delta_text`, the token
        // count, and the finish reason are byte-identical to the pre-F4 loop.
        // The matcher is only consulted in the ANSWER phase, mirroring the wire
        // layer (stop sequences bound the answer, never the `<think>` trace);
        // on the thinking-off default path the loop is always in the answer
        // phase so this is a no-op distinction.
        let mut stop_matcher = StopMatcher::new(request.stop_text.clone());
        // `Stop` is the value only for the degenerate `max_tokens == 0` answer
        // budget (the entry guard breaks emitting nothing — byte-identical to
        // the pre-Part-4 `for _ in 0..0`). Every other exit reassigns it.
        let mut finish_reason = FinishReason::Stop;

        // -- Part 4: reasoning-phase state machine + forced-close -----------
        //
        // The decode loop runs in one of two phases:
        //   * Reasoning  — inside the `<think>` block (only entered when the
        //                  request resolved `enable_thinking == true`).
        //   * Answer     — after `</think>` (natural OR forced).
        // The ANSWER budget (`request.max_tokens`) bounds ONLY answer tokens,
        // so a long reasoning trace can never starve the answer. The reasoning
        // trace is bounded SEPARATELY by `request.reasoning_budget`; on
        // overrun the loop FORCE-CLOSES by injecting `</think>\n\n` ids
        // (advancing the KV via `session.extend`) and switches to the answer
        // phase.
        //
        // THINKING-OFF byte-identity: when `enable_thinking` is false the
        // phase starts at `Answer`, the detector / forced-close are never
        // consulted (both gated on the reasoning phase), and the answer budget
        // is the SAME post-emission `>= max_tokens` bound the pre-Part-4 loop
        // used. The emitted token stream, count, and finish reason are
        // therefore identical. (Guarded by `tests/reasoning_budget_test.rs`.)
        let mut phase = if request.enable_thinking {
            ReasoningPhase::Reasoning
        } else {
            ReasoningPhase::Answer
        };
        let mut reasoning_generated = 0usize;
        let mut answer_generated = 0usize;
        // Detects the model-emitted `</think>` close. Resolves the marker's
        // token id(s) from the tokenizer ONCE (lazily) and matches by id when
        // the tokenizer encodes `</think>` as a single special token (the
        // Qwen3.5 production case), falling back to a straddle-safe text scan
        // when it does not (e.g. a byte-level tokenizer). Only consulted in the
        // reasoning phase.
        let mut close_detector = ThinkCloseDetector::new();
        // Forced-close injection ids, computed lazily so the thinking-off path
        // never tokenizes the marker.
        let mut close_ids: Option<Vec<u32>> = None;

        loop {
            // Check cancel BEFORE running the decode
            // kernel.  A long max_tokens (e.g. max_tokens=10000)
            // would otherwise burn one full token's worth of GPU time
            // after every cancel observation.  The pre-check costs one
            // atomic load — negligible vs the decode work it gates.
            //
            // Reconcile with decode-delay
            // (`LUMEN_CUDA_DECODE_DELAY_US`): the optional CUDA sleep
            // lives INSIDE `session.next_token` (cuda backend impl).
            // Because we check `cancel` here BEFORE entering
            // `next_token`, the worst-case latency to observe a
            // cancel during a long generation is one decode step +
            // any configured delay — bounded by the delay setting
            // (default 0, max documented 50µs).  The cancel check
            // dominates the delay so the wedge-detection bound stays tight.
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            // Degenerate-answer-budget entry guard: in the answer phase with
            // the budget already met and nothing emitted (the `max_tokens == 0`
            // case, firing on iteration 0 with `answer_generated == 0`), fall
            // out emitting nothing — byte-identical to the pre-Part-4
            // `for _ in 0..0` (`{stop, completion=0}`). For `max_tokens >= 1`
            // this never fires at entry (post-emission bound enforces it).
            if phase == ReasoningPhase::Answer && answer_generated >= request.max_tokens {
                break;
            }
            // FORCED-CLOSE: reasoning budget exhausted while still reasoning.
            // Inject `</think>\n\n` (advances KV so decode resumes past the
            // closed block), emit those bytes, switch to the answer phase.
            // Gated on the reasoning phase, so never fires when thinking off.
            if phase == ReasoningPhase::Reasoning
                && request.reasoning_budget > 0
                && reasoning_generated >= request.reasoning_budget
            {
                let ids = close_ids.get_or_insert_with(|| self.tokenizer.encode("</think>\n\n"));
                if !ids.is_empty() {
                    if session.kv().seq_len() + ids.len() > self.config.max_seq_len {
                        finish_reason = FinishReason::Length;
                        break;
                    }
                    match session.extend(ids, self.backend.as_ref(), self.weights.as_ref()) {
                        Ok(_) => {
                            let mut inject_text = String::new();
                            for &id in ids.iter() {
                                inject_text.push_str(
                                    &self.tokenizer.decode_incremental(&mut bytes_state, id),
                                );
                            }
                            if !inject_text.is_empty()
                                && self
                                    .send_event_polling_cancel(
                                        &tokens_tx,
                                        &cancel,
                                        TokenEvent::Token {
                                            token_id: *ids.last().unwrap(),
                                            delta_text: inject_text,
                                        },
                                    )
                                    .is_err()
                            {
                                return;
                            }
                        }
                        Err(e) => {
                            if matches!(&e, RuntimeError::KvCache(msg) if msg.contains("would exceed max_seq_len"))
                            {
                                finish_reason = FinishReason::Length;
                                break;
                            }
                            let _ = self.send_event_polling_cancel(
                                &tokens_tx,
                                &cancel,
                                TokenEvent::Error(format!("forced-close inject: {e}")),
                            );
                            return;
                        }
                    }
                }
                phase = ReasoningPhase::Answer;
            }
            // Proactive KV-overflow check BEFORE
            // dispatching the decode kernel.
            //
            // `kv.advance_seq_len()` (and its CUDA batched twin
            // `advance_seq_len_by`) error with
            // `RuntimeError::KvCache("sequence length X would
            // exceed max_seq_len Y")` when the next token would
            // push past the configured context window.  Without
            // the proactive check, that runtime error bubbles out
            // of `session.next_token`, gets translated to
            // `TokenEvent::Error("decode: ...")`, and the wire
            // layer renders it as HTTP 500 (non-streaming) or an
            // inline `{"error":...}` SSE frame followed by
            // `[DONE]` — neither of which matches the OpenAI
            // ChatCompletion spec, which requires
            // `finish_reason: "length"` when the model hits the
            // context window mid-decode.
            //
            // The proactive check has three advantages over
            // catching the runtime error after the fact:
            //
            // 1. NO GPU work wasted.  GPU decode paths write KV
            //    state to device memory FIRST and then call the
            //    CPU-side `advance_seq_len()` for tracking; a
            //    post-call match would have already burned one
            //    decode's worth of FLOPs and DMA.
            // 2. NO partial-state risk.  Some backends might
            //    leave per-layer scratch buffers (RoPE tables,
            //    attention scores) in an inconsistent state when
            //    the post-kernel CPU tracking fails.  By checking
            //    BEFORE the dispatch we never touch the kernels
            //    in the overflow case.
            // 3. Single observation site.  The same check covers
            //    the CPU naive backend (whose `next_token` path
            //    calls `kv.advance_seq_len()` directly in
            //    `session.rs::next_token`) AND every GPU backend
            //    (where `advance_seq_len` is called inside
            //    `decode_token` / `decode_token_greedy`).
            //
            // The prompt-length guard above (line ~1304) handles
            // the case where the prompt ALONE exceeds max_seq_len;
            // that remains a `TokenEvent::Error` (and HTTP 400-
            // shaped on the client) because it is a client-side
            // input-too-large condition, not a generate-until-
            // context-full condition.
            if session.kv().seq_len() + 1 > self.config.max_seq_len {
                finish_reason = FinishReason::Length;
                break;
            }
            let res = session.next_token(self.backend.as_ref(), self.weights.as_ref());
            let token_id = match res {
                Ok(id) => id,
                Err(e) => {
                    // belt-and-suspenders: even with the
                    // proactive check above, treat any KvCache-
                    // shaped overflow error as a clean `Length`
                    // termination rather than a mid-stream 500.
                    // This covers backend paths whose KV-overflow
                    // accounting might evolve independently of the
                    // shared `kv.advance_seq_len()` check (e.g. a
                    // backend that pre-validates differently or a
                    // future per-layer KV cap).  The sentinel
                    // substring `"would exceed max_seq_len"`
                    // matches the formatter in
                    // `kv::mod.rs::advance_seq_len`; any other
                    // KvCache error (e.g. layer-index, view-state)
                    // remains an HTTP-500-shaped `Error`.
                    if matches!(&e, RuntimeError::KvCache(msg) if msg.contains("would exceed max_seq_len"))
                    {
                        finish_reason = FinishReason::Length;
                        break;
                    }
                    let _ = self.send_event_polling_cancel(
                        &tokens_tx,
                        &cancel,
                        TokenEvent::Error(format!("decode: {e}")),
                    );
                    return;
                }
            };
            generated += 1;
            // Charge the token to the active phase. A token that carries (or
            // completes) `</think>` is the last reasoning token; the answer
            // begins on the following token.
            match phase {
                ReasoningPhase::Reasoning => reasoning_generated += 1,
                ReasoningPhase::Answer => answer_generated += 1,
            }

            // EOS check (token-id based).
            if request.eos_token_ids.contains(&token_id) {
                // Emit the residual decoded text but NOT the EOS token text.
                let _ = self.tokenizer.decode_incremental(&mut bytes_state, token_id);
                finish_reason = FinishReason::Stop;
                break;
            }

            let delta = self
                .tokenizer
                .decode_incremental(&mut bytes_state, token_id);

            // Phase transition: detect the model-emitted `</think>` close by
            // token id (special-token fast path) or straddle-safe text scan
            // (fallback). Only while reasoning. The FULL `delta` is observed so
            // the close detector sees every byte even when the stop matcher
            // (below) holds part of it back.
            if phase == ReasoningPhase::Reasoning
                && close_detector.observe(token_id, &delta, self.tokenizer.as_ref())
            {
                phase = ReasoningPhase::Answer;
            }

            // F4 textual stop. ONLY engaged when the matcher is active (a
            // non-empty `stop_text`) AND we are in the answer phase. The
            // EMPTY-STOP / reasoning-phase path takes the `else` branch which
            // emits `delta` verbatim — byte-identical to the pre-F4 loop
            // (including empty deltas from partial-UTF-8 tokens, which the
            // matcher branch would otherwise suppress).
            if stop_matcher.is_active() && phase == ReasoningPhase::Answer {
                let (safe, hit_stop) = stop_matcher.push(&delta);
                // Emit the safe prefix (the matcher holds back any tail that
                // could still complete a stop sequence). Skip empty fragments —
                // the wire layer suppresses empty-content frames anyway, and an
                // active stop matcher is never on the byte-identity path.
                if !safe.is_empty()
                    && self
                        .send_event_polling_cancel(
                            &tokens_tx,
                            &cancel,
                            TokenEvent::Token {
                                token_id,
                                delta_text: safe,
                            },
                        )
                        .is_err()
                {
                    return;
                }
                if hit_stop {
                    // The matched stop bytes (and anything after) are dropped
                    // per stop semantics; the prefix before it was already
                    // emitted above. Report StopSequence (OpenAI "stop" /
                    // Anthropic "stop_sequence") to distinguish a caller stop
                    // string from a natural EOS end-of-turn.
                    finish_reason = FinishReason::StopSequence;
                    break;
                }
            } else if self
                .send_event_polling_cancel(
                    &tokens_tx,
                    &cancel,
                    TokenEvent::Token {
                        token_id,
                        delta_text: delta,
                    },
                )
                .is_err()
            {
                return;
            }

            // ANSWER budget: bounds ONLY answer-phase tokens. On the
            // thinking-off path `answer_generated == generated`, so this is the
            // SAME stop as the pre-Part-4 loop.
            if phase == ReasoningPhase::Answer && answer_generated >= request.max_tokens {
                finish_reason = FinishReason::Length;
                break;
            }
        }

        // F4 residual flush: drain any bytes the stop matcher held back across
        // the final iteration (a partial stop-prefix that the natural EOS /
        // max-tokens / KV-overflow termination never completed — so those bytes
        // are genuine answer content and safe to emit). On a `hit_stop` exit the
        // matcher's window was already cleared, so this yields "". On the
        // EMPTY-STOP path `finish()` always yields "" (the matcher never holds
        // anything), so NO event is emitted and the stream stays byte-identical.
        let residual_stop = stop_matcher.finish();
        if !residual_stop.is_empty() {
            // The held tail belongs to the last decoded token; report its id for
            // logging parity. The wire layer keys only on `delta_text`.
            let _ = self.send_event_polling_cancel(
                &tokens_tx,
                &cancel,
                TokenEvent::Token {
                    token_id: 0,
                    delta_text: residual_stop,
                },
            );
        }

        let _ = self.send_event_polling_cancel(
            &tokens_tx,
            &cancel,
            TokenEvent::Done {
                finish_reason,
                prompt_tokens: request.prompt_tokens.len(),
                completion_tokens: generated,
            },
        );

        // Suppress unused-warning for prior_tokens; we keep the read in
        // case future logging instrumentation wants it.
        let _ = prior_tokens;
    }
}

// -- helpers --

/// Decode-loop reasoning phase (Part 4). Starts at `Reasoning` only when the
/// request resolved `enable_thinking == true`; otherwise the loop runs
/// entirely in `Answer`, which is byte-identical to the pre-Part-4 behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningPhase {
    Reasoning,
    Answer,
}

/// Detects the model-emitted `</think>` close marker during decode so the loop
/// can flip from the reasoning phase to the answer phase.
///
/// Detection is hybrid: it resolves the marker's token id(s) ONCE from the
/// tokenizer and, when `</think>` encodes to a SINGLE id (the Qwen3.5
/// special-token case), matches by that id — exact and free. When the
/// tokenizer does not have `</think>` as one token (e.g. a byte-level
/// tokenizer), it falls back to a straddle-safe scan of the decoded text using
/// the same `tooling::THINK_CLOSE` constant the wire-layer `ReasoningExtractor`
/// splits on, so the decode-loop boundary matches the downstream split.
struct ThinkCloseDetector {
    /// The marker's token id IFF `</think>` encodes to exactly one token.
    /// Resolved lazily on first `observe` (so the thinking-off path never
    /// tokenizes it). `None` until resolved; `Some(None)` = resolved to "not a
    /// single token" (use the text fallback); `Some(Some(id))` = match by id.
    marker_id: Option<Option<u32>>,
    /// Trailing bytes from the previous fragment that could begin the marker
    /// (text-fallback carry). At most `THINK_CLOSE.len() - 1`.
    carry: String,
}

impl ThinkCloseDetector {
    fn new() -> Self {
        Self { marker_id: None, carry: String::new() }
    }

    /// Observe one decoded token (its id + decoded fragment). Returns `true`
    /// the first time `</think>` is recognised.
    fn observe(&mut self, token_id: u32, fragment: &str, tok: &dyn Tokenize) -> bool {
        // Resolve the single-token marker id on first use.
        let marker_id = *self.marker_id.get_or_insert_with(|| {
            let ids = tok.encode(lumen_runtime::tooling::THINK_CLOSE);
            if ids.len() == 1 {
                Some(ids[0])
            } else {
                None
            }
        });
        // Fast path: exact special-token id.
        if let Some(id) = marker_id {
            if token_id == id {
                return true;
            }
            // Even with a single-token marker, also run the text fallback in
            // case the model spelled the marker out as separate sub-tokens.
        }
        // Text fallback: straddle-safe scan.
        let marker = lumen_runtime::tooling::THINK_CLOSE;
        let mut buf = String::with_capacity(self.carry.len() + fragment.len());
        buf.push_str(&self.carry);
        buf.push_str(fragment);
        self.carry.clear();
        if buf.contains(marker) {
            return true;
        }
        let keep = marker_prefix_overlap(buf.as_bytes(), marker.as_bytes());
        if keep > 0 {
            self.carry.push_str(&buf[buf.len() - keep..]);
        }
        false
    }
}

/// Length of the longest suffix of `tail` that is a prefix of `marker`
/// (capped at `marker.len() - 1`). Pure helper for the text fallback.
fn marker_prefix_overlap(tail: &[u8], marker: &[u8]) -> usize {
    let max = tail.len().min(marker.len().saturating_sub(1));
    for k in (1..=max).rev() {
        if tail[tail.len() - k..] == marker[..k] {
            return k;
        }
    }
    0
}

/// Best-effort extraction of a human-readable message from a
/// `Box<dyn Any + Send>` panic payload as returned by
/// `std::panic::catch_unwind`.
///
/// `panic!()` payloads from the standard library are typically `&str`
/// or `String`; third-party `unwrap()`-induced panics (e.g. cudarc
/// `DriverError` displays via `Debug`) come through as a `String`.
/// Anything more exotic is reported as `"<non-string panic payload>"`
/// so the recovery path never panics itself trying to format the
/// original panic.
fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "<non-string panic payload>".to_string()
}

// -- helpers --

/// Compute the recursive byte size of a directory tree.  Used by the
/// post-job memory-breakdown update to track disk-KV residency.
///
/// Returns 0 on any I/O error (the soak harness already tracks this via
/// its own recursive walker; this helper exists so the breakdown is
/// self-contained when read in isolation).  Symbolic links are not
/// followed — the disk-KV layout writes plain files only.
fn dir_size_recursive(dir: &std::path::Path) -> u64 {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_file() {
            total = total.saturating_add(meta.len());
        } else if meta.is_dir() {
            total = total.saturating_add(dir_size_recursive(&entry.path()));
        }
    }
    total
}

/// Best-effort Tokio "alive task" count for the current runtime.
///
/// Returns 0 if no Tokio runtime is reachable from the calling thread
/// (the breakdown is unaware of inactive tasks).  The
/// `Handle::current().metrics()` call requires `tokio_unstable`-enabled
/// builds OR Tokio ≥ 1.34 stable `num_alive_tasks` support — Lumen ships
/// against Tokio 1.x stable, where `num_alive_tasks` IS stable for the
/// multi-thread runtime.  When the metric is not surfaced we silently
/// return 0 so the breakdown reports 0 rather than panicking the worker.
fn current_tokio_alive_tasks() -> u64 {
    // `Handle::try_current` returns `Err` when called from a
    // non-runtime thread.  Our worker runs on `spawn_blocking`, which IS
    // inside the runtime, so the handle is normally reachable.
    match tokio::runtime::Handle::try_current() {
        Ok(h) => h.metrics().num_alive_tasks() as u64,
        Err(_) => 0,
    }
}

// -- Convenience adapters --

/// Minimal tokenizer for tests / synthetic models: maps each input byte
/// to its own token id and vice versa, with EOS = 0.
pub struct IdentityByteTokenizer {
    eos: Vec<u32>,
}

impl Default for IdentityByteTokenizer {
    fn default() -> Self {
        Self { eos: vec![0] }
    }
}

impl IdentityByteTokenizer {
    pub fn new(eos: Vec<u32>) -> Self {
        Self { eos }
    }
}

impl Tokenize for IdentityByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }
    fn decode_incremental(&self, state: &mut Vec<u8>, token_id: u32) -> String {
        // Byte tokens may correspond to partial UTF-8 codepoints; buffer
        // bytes and only flush complete codepoints.
        state.push((token_id & 0xff) as u8);
        // Try to take the longest UTF-8 prefix.
        match std::str::from_utf8(state) {
            Ok(_) => {
                let s = String::from_utf8(std::mem::take(state)).unwrap_or_default();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid == 0 {
                    String::new()
                } else {
                    let head = state[..valid].to_vec();
                    let tail = state[valid..].to_vec();
                    *state = tail;
                    String::from_utf8(head).unwrap_or_default()
                }
            }
        }
    }
    fn eos_tokens(&self) -> Vec<u32> {
        self.eos.clone()
    }
}

// =========================================================================
// Channel-pool unit tests.
//
// These tests exercise the [`PooledReceiver`] / channel-pool recycling
// behaviour in isolation, without spinning up an EngineWorker.  They cover:
//
// 1. Pool returns the pair on receiver drop (steady state).
// 2. Drain on drop: stale events from a previous job do not leak into
//    the next user.
// 3. Pool cap: returns above the cap are dropped (no unbounded growth).
// 4. Overflow path: when the pool is empty, `take_channel_pair` allocates
//    a fresh pair and the drop does NOT recycle (pool stays empty).
// 5. Channel-alive invariant: pool's retained Sender keeps the channel
//    open even after the worker's clone is dropped.
//
// =========================================================================
#[cfg(test)]
mod pool_tests {
    use super::*;

    /// Helper: build a channel pool pre-populated with `n` pairs of
    /// `POOL_CHANNEL_CAPACITY` slots.  Returns the pool plus its cap.
    fn make_pool(n: usize, cap: usize) -> (ChannelPool, usize) {
        let mut deque = VecDeque::with_capacity(cap);
        for _ in 0..n {
            deque.push_back(mpsc::channel::<TokenEvent>(POOL_CHANNEL_CAPACITY));
        }
        (Arc::new(Mutex::new(deque)), cap)
    }

    /// Helper: take a pair from the pool, mirroring
    /// `EngineHandle::take_channel_pair` minus the fall-back-to-fresh
    /// branch.  Returns None when the pool is empty (so tests can assert
    /// on it directly).
    fn pop_pair(pool: &ChannelPool) -> Option<(mpsc::Sender<TokenEvent>, mpsc::Sender<TokenEvent>, mpsc::Receiver<TokenEvent>)> {
        let mut guard = pool.lock().unwrap();
        guard.pop_front().map(|(return_sender, rx)| {
            let worker_tx = return_sender.clone();
            (worker_tx, return_sender, rx)
        })
    }

    /// Pool returns the (sender, receiver) pair when the PooledReceiver
    /// is dropped after a normal-shaped exchange.
    #[tokio::test]
    async fn pool_recycles_pair_on_drop() {
        let (pool, cap) = make_pool(1, 2);
        assert_eq!(pool.lock().unwrap().len(), 1);

        // Pop and use the pair.
        let (worker_tx, return_sender, rx) = pop_pair(&pool).expect("pool primed");
        assert_eq!(pool.lock().unwrap().len(), 0);

        let mut pooled = PooledReceiver::new(rx, return_sender, Some(Arc::clone(&pool)), cap, None);

        // Simulate a one-event exchange.
        worker_tx
            .send(TokenEvent::Done {
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 1,
            })
            .await
            .unwrap();
        let evt = pooled.recv().await.expect("event");
        assert!(matches!(evt, TokenEvent::Done { .. }));

        // Worker drops its clone (mimic process_job exit).
        drop(worker_tx);

        // Receiver-handler exits scope -> PooledReceiver dropped -> pool returns to 1.
        drop(pooled);
        assert_eq!(pool.lock().unwrap().len(), 1, "pool must have recycled the pair");
    }

    /// PooledReceiver::drop drains any leftover events from the channel
    /// before recycling, so the next user gets a clean buffer.
    #[tokio::test]
    async fn pool_drains_leftover_events_before_recycling() {
        let (pool, cap) = make_pool(1, 2);
        let (worker_tx, return_sender, rx) = pop_pair(&pool).expect("pool primed");

        let mut pooled = PooledReceiver::new(rx, return_sender, Some(Arc::clone(&pool)), cap, None);

        // Worker sends three events; receiver consumes only one.
        for _ in 0..3 {
            worker_tx
                .send(TokenEvent::Token {
                    token_id: 0,
                    delta_text: "stale".into(),
                })
                .await
                .unwrap();
        }
        let _consumed = pooled.recv().await.expect("event 1");

        // Drop the worker clone and the receiver.  Two leftover events
        // remain in the channel; drop must drain them.
        drop(worker_tx);
        drop(pooled);

        // Pop the recycled pair and verify the receiver is clean.
        let (_worker_tx, _return_sender, mut rx) =
            pop_pair(&pool).expect("recycled pair present");
        match rx.try_recv() {
            Err(mpsc::error::TryRecvError::Empty)
            | Err(mpsc::error::TryRecvError::Disconnected) => {
                // empty channel — drain succeeded.
            }
            Ok(evt) => panic!("expected empty channel, got stale event {:?}", evt),
        }
    }

    /// Pool cap: returns above the cap are dropped, not recycled.
    #[tokio::test]
    async fn pool_cap_drops_overflow_returns() {
        // Cap = 1; start with 0 in pool.
        let (pool, cap) = make_pool(0, 1);

        // Manually push 1 pair (fill to cap).
        {
            let mut g = pool.lock().unwrap();
            g.push_back(mpsc::channel(POOL_CHANNEL_CAPACITY));
        }
        assert_eq!(pool.lock().unwrap().len(), 1);

        // Now build a PooledReceiver with an EXTRA pair (not in pool).
        let (extra_tx, extra_rx) = mpsc::channel::<TokenEvent>(POOL_CHANNEL_CAPACITY);
        let return_sender = extra_tx.clone();
        let worker_tx = extra_tx.clone();
        let pooled = PooledReceiver::new(extra_rx, return_sender, Some(Arc::clone(&pool)), cap, None);

        // Drop everything; the pool should NOT grow above cap.
        drop(worker_tx);
        drop(extra_tx);
        drop(pooled);

        assert_eq!(
            pool.lock().unwrap().len(),
            1,
            "pool must remain at cap; overflow return dropped"
        );
    }

    /// Unpooled receiver (pool=None) does NOT panic on drop and does
    /// NOT attempt to recycle.  Used by `collect_chat_from_events` for
    /// the test-only path.
    #[tokio::test]
    async fn unpooled_receiver_drops_safely() {
        let (tx, rx) = mpsc::channel::<TokenEvent>(POOL_CHANNEL_CAPACITY);
        let return_sender = tx.clone();
        let mut pooled = PooledReceiver::new(rx, return_sender, None, 0, None);

        tx.send(TokenEvent::Done {
            finish_reason: FinishReason::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        })
        .await
        .unwrap();
        let _ = pooled.recv().await;

        drop(pooled); // No pool to return to; must not panic.
    }

    /// Channel-alive invariant: the pool's retained Sender keeps the
    /// channel open across a request, even after the worker's clone is
    /// dropped (which happens at process_job exit).  This is the
    /// load-bearing guarantee that lets us pool channels at all.
    #[tokio::test]
    async fn pool_retained_sender_keeps_channel_alive() {
        let (pool, cap) = make_pool(1, 2);
        let (worker_tx, return_sender, rx) = pop_pair(&pool).expect("pool primed");

        // Wrap rx + return_sender.  pool=Some so drop will recycle.
        let mut pooled = PooledReceiver::new(rx, return_sender, Some(Arc::clone(&pool)), cap, None);

        // Worker drops its clone IMMEDIATELY without sending anything.
        drop(worker_tx);

        // The channel must NOT be reported as disconnected — the pool's
        // retained sender (held inside `pooled`) keeps it alive.
        match pooled.try_recv() {
            Err(mpsc::error::TryRecvError::Empty) => {
                // expected: channel is empty but NOT disconnected.
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {
                panic!("channel closed despite pool's retained sender");
            }
            Ok(evt) => panic!("expected empty, got {:?}", evt),
        }

        drop(pooled);
        assert_eq!(
            pool.lock().unwrap().len(),
            1,
            "pair must recycle even after worker dropped its clone"
        );
    }

    /// take_channel_pair (the EngineHandle path) returns a fresh pair
    /// when the pool is empty (overflow), and the resulting
    /// PooledReceiver does NOT recycle (pool stays empty).
    #[tokio::test]
    async fn overflow_path_does_not_grow_pool() {
        use crate::engine::EngineHandle;

        // Build an EngineHandle manually without spawning a worker, just
        // to exercise `take_channel_pair`'s overflow branch.
        let (sender, _job_inbox_rx) = mpsc::channel::<WorkerJob>(1);
        let breakdown = Arc::new(Mutex::new(ServerMemoryBreakdown::default()));
        let channel_pool: ChannelPool = Arc::new(Mutex::new(VecDeque::new())); // empty pool
        let handle = EngineHandle {
            sender,
            model_info: Arc::new(ModelInfo {
                id: "test".into(),
                owned_by: "test".into(),
                created: 0,
                context_length: 1,
            }),
            tokenizer: Arc::new(IdentityByteTokenizer::default()) as Arc<dyn Tokenize>,
            breakdown,
            channel_pool: Arc::clone(&channel_pool),
            pool_cap: 2,
        };

        assert_eq!(handle.channel_pool_len(), 0, "pool starts empty");
        let (_tx, _rx, _return_sender, pool_handle) = handle.take_channel_pair(8);
        assert!(
            pool_handle.is_none(),
            "overflow allocation must NOT carry a pool handle"
        );
        // Even after dropping the PooledReceiver, the pool stays empty.
        // (The PooledReceiver's pool=None means drop doesn't push back.)
        let pooled =
            PooledReceiver::new(_rx, _return_sender, pool_handle, handle.pool_cap, None);
        drop(pooled);
        assert_eq!(
            handle.channel_pool_len(),
            0,
            "overflow path must not grow pool on drop"
        );
    }

    /// Stress: repeated take + drop must NEVER allocate beyond cap.
    /// This is the steady-state behaviour under load — the whole point
    /// of the pool is that high request rates do not balloon RSS.
    #[tokio::test]
    async fn steady_state_pool_size_is_bounded() {
        let (pool, cap) = make_pool(16, 17); // 16 primed, cap 17.

        for _ in 0..1000 {
            let (worker_tx, return_sender, rx) = pop_pair(&pool).expect("pool primed");
            let mut pooled =
                PooledReceiver::new(rx, return_sender, Some(Arc::clone(&pool)), cap, None);
            // Quick exchange.
            worker_tx
                .send(TokenEvent::Done {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: 1,
                    completion_tokens: 1,
                })
                .await
                .unwrap();
            let _ = pooled.recv().await;
            drop(worker_tx);
            drop(pooled);
        }

        assert!(
            pool.lock().unwrap().len() <= cap,
            "pool grew beyond cap after 1000 iterations"
        );
        // Specifically, depth should be exactly 16 (every iter returned 1 pair).
        assert_eq!(
            pool.lock().unwrap().len(),
            16,
            "steady-state pool depth must match primed count"
        );
    }
}

// =========================================================================
// panic-supervisor helper unit tests.
//
// The end-to-end recovery behaviour of the worker loop is covered by the
// integration test in `tests/server_integration.rs` (driving a backend
// that panics on the N-th call).  These small unit tests pin the
// `panic_payload_message` extraction contract so the recovery branch
// can format any payload safely.
// =========================================================================
#[cfg(test)]
mod panic_supervisor_tests {
    use super::*;
    use std::panic;

    #[test]
    fn payload_message_extracts_str_literal() {
        let res = panic::catch_unwind(|| panic!("static str payload"));
        let payload = res.unwrap_err();
        assert_eq!(panic_payload_message(payload.as_ref()), "static str payload");
    }

    #[test]
    fn payload_message_extracts_formatted_string() {
        let res = panic::catch_unwind(|| panic!("formatted {}", 42));
        let payload = res.unwrap_err();
        assert_eq!(panic_payload_message(payload.as_ref()), "formatted 42");
    }

    #[test]
    fn payload_message_handles_non_string_payload() {
        // Custom payload type that is `Any + Send` but not a string —
        // mirrors what a panicking third-party crate could throw.
        struct Custom(#[allow(dead_code)] u32);
        let res = panic::catch_unwind(|| panic::panic_any(Custom(7)));
        let payload = res.unwrap_err();
        assert_eq!(
            panic_payload_message(payload.as_ref()),
            "<non-string panic payload>"
        );
    }

    #[test]
    fn payload_message_handles_unwrap_err_string() {
        // Simulate the cudarc panic shape: `Result::unwrap` on `Err`
        // formats via `Debug`, so the payload is a `String`.
        let err: Result<(), String> = Err("DriverError(CUDA_ERROR_OUT_OF_MEMORY)".into());
        let res = panic::catch_unwind(|| err.unwrap());
        let payload = res.unwrap_err();
        let msg = panic_payload_message(payload.as_ref());
        assert!(
            msg.contains("CUDA_ERROR_OUT_OF_MEMORY"),
            "unwrap payload should mention the inner error, got: {msg}"
        );
    }

    #[test]
    fn panic_window_constants_are_sane() {
        // Guard against accidental zeroing during a refactor — these
        // values determine production recovery semantics.
        assert!(
            DEFAULT_PANIC_WINDOW_SECS >= 1,
            "DEFAULT_PANIC_WINDOW_SECS must be > 0"
        );
        assert!(
            DEFAULT_MAX_PANICS_IN_WINDOW >= 1,
            "DEFAULT_MAX_PANICS_IN_WINDOW must be >= 1"
        );
    }

    #[test]
    fn panic_window_env_override_parses() {
        // Smoke-test the env resolvers: a malformed value falls back
        // to the default; a well-formed value parses through.  This
        // is the same behaviour the worker uses at spawn time.
        std::env::set_var("LUMEN_SERVER_PANIC_WINDOW_SECS", "5");
        std::env::set_var("LUMEN_SERVER_PANIC_MAX", "10");
        assert_eq!(panic_window().as_secs(), 5);
        assert_eq!(max_panics_in_window(), 10);
        std::env::set_var("LUMEN_SERVER_PANIC_WINDOW_SECS", "not-a-number");
        std::env::set_var("LUMEN_SERVER_PANIC_MAX", "");
        assert_eq!(panic_window().as_secs(), DEFAULT_PANIC_WINDOW_SECS);
        assert_eq!(max_panics_in_window(), DEFAULT_MAX_PANICS_IN_WINDOW);
        std::env::remove_var("LUMEN_SERVER_PANIC_WINDOW_SECS");
        std::env::remove_var("LUMEN_SERVER_PANIC_MAX");
        assert_eq!(panic_window().as_secs(), DEFAULT_PANIC_WINDOW_SECS);
        assert_eq!(max_panics_in_window(), DEFAULT_MAX_PANICS_IN_WINDOW);
    }
}

// =========================================================================
// Part 4: `ThinkCloseDetector` unit tests.
//
// Cover both detection paths: the special-token id fast path (Qwen3.5
// production tokeniser) and the straddle-safe text fallback (byte-level
// tokeniser). The end-to-end budget / forced-close behaviour is covered in
// `tests/reasoning_budget_test.rs`.
// =========================================================================
#[cfg(test)]
mod think_close_detector_tests {
    use super::*;

    /// Tokeniser that encodes `</think>` as ONE special id (id 7), everything
    /// else as bytes — mirrors the real Qwen3.5 special-token registration.
    struct SpecialThinkTokenizer;
    impl Tokenize for SpecialThinkTokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            if text == lumen_runtime::tooling::THINK_CLOSE {
                vec![7]
            } else {
                text.bytes().map(|b| b as u32).collect()
            }
        }
        fn decode_incremental(&self, _s: &mut Vec<u8>, id: u32) -> String {
            if id == 7 { "</think>".to_string() } else { ((id & 0xff) as u8 as char).to_string() }
        }
        fn eos_tokens(&self) -> Vec<u32> { vec![0] }
    }

    /// Byte tokeniser: `</think>` is NOT a single token (forces the text
    /// fallback). `encode("</think>")` returns 8 byte ids.
    struct ByteTokenizer;
    impl Tokenize for ByteTokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            text.bytes().map(|b| b as u32).collect()
        }
        fn decode_incremental(&self, _s: &mut Vec<u8>, id: u32) -> String {
            ((id & 0xff) as u8 as char).to_string()
        }
        fn eos_tokens(&self) -> Vec<u32> { vec![0] }
    }

    #[test]
    fn special_token_id_fast_path() {
        let tok = SpecialThinkTokenizer;
        let mut d = ThinkCloseDetector::new();
        // A normal token does not match.
        assert!(!d.observe(104, "h", &tok));
        // The special `</think>` id matches immediately.
        assert!(d.observe(7, "</think>", &tok));
    }

    #[test]
    fn byte_tokenizer_text_fallback_one_fragment() {
        let tok = ByteTokenizer;
        let mut d = ThinkCloseDetector::new();
        // Feed the whole marker as one (multi-byte) fragment via a single
        // observe; the id won't match (no single-token marker) but the text
        // scan catches it.
        assert!(d.observe(62, "reasoning</think>", &tok));
    }

    #[test]
    fn byte_tokenizer_text_fallback_split() {
        let tok = ByteTokenizer;
        let mut d = ThinkCloseDetector::new();
        assert!(!d.observe(105, "</thi", &tok));
        assert!(d.observe(62, "nk>", &tok));
    }

    #[test]
    fn special_tokenizer_also_catches_spelled_out_marker() {
        // Even when `</think>` has a single id, a model that spells it out as
        // separate sub-tokens is still caught via the text fallback.
        let tok = SpecialThinkTokenizer;
        let mut d = ThinkCloseDetector::new();
        // Decoded text spells the marker without ever using id 7.
        assert!(d.observe(62, "thinking</think>", &tok));
    }

    #[test]
    fn near_miss_is_not_a_false_positive() {
        let tok = ByteTokenizer;
        let mut d = ThinkCloseDetector::new();
        assert!(!d.observe(105, "</thi", &tok));
        assert!(!d.observe(110, "nking", &tok));
    }

    #[test]
    fn marker_prefix_overlap_basic() {
        let m = b"</think>";
        assert_eq!(marker_prefix_overlap(b"x</thi", m), 5);
        assert_eq!(marker_prefix_overlap(b"x<", m), 1);
        assert_eq!(marker_prefix_overlap(b"xyz", m), 0);
        assert_eq!(marker_prefix_overlap(b"</think>", m), 0); // full match handled by contains
    }
}
