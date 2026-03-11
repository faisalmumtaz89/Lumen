//! Hand-rolled, zero-dependency persistent thread pool for parallel matmul.
//!
//! Design goals:
//! - Zero allocation on the hot path (all buffers pre-allocated)
//! - Hybrid spin-then-park wake: workers spin on an atomic generation counter
//!   to catch fast dispatches (~100-200 us apart during inference), falling back
//!   to condvar sleep when idle to save CPU
//! - Persistent worker threads created once, parked between calls
//! - Configurable thread count (default = available parallelism - 1)
//! - Thread-safe: `Send + Sync` for use from `&self` compute methods
//!
//! Workers are created at construction and live until the pool is dropped.
//! `parallel_for` wakes them, they do their chunk, signal completion, and sleep.
//! This eliminates the ~1-5us per-call spawn/join overhead of `std::thread::scope`
//! (154 spawn/join cycles per token with 22 layers x 7 matmuls).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Minimum number of rows per thread to amortize thread dispatch overhead.
/// Lowered from 64 to 32 to improve load balancing on heterogeneous cores
/// (Apple M4: 4P + 6E). With 32, a 2048-row matmul still gets 2048/10 = 204
/// rows per thread, well above this floor. The lower threshold helps when
/// fewer rows are available (e.g., 512-row ops use all 10 threads: 512/10=51 >= 32).
const MIN_ROWS_PER_THREAD: usize = 32;

/// Minimum out_dim to consider parallelization. Below this threshold,
/// the single-threaded path is used unconditionally to avoid overhead
/// on small ops like attention dot products (head_dim=64).
const PARALLEL_THRESHOLD: usize = 256;

/// Maximum number of spin iterations before falling back to condvar wait.
/// During inference, parallel_for fires every ~100-200 us. 128 spins of
/// `spin_loop` hint (~0.5-1 us total on modern hardware) is enough to catch
/// most dispatches without mutex acquisition, while keeping CPU waste minimal
/// when the engine is idle between tokens.
const SPIN_LIMIT: u32 = 128;

/// Number of "fast" (Performance) cores for asymmetric work partitioning.
/// On Apple M4: 4 P-cores + 6 E-cores. P-cores are ~1.5x faster for
/// compute-bound matmul work. By giving P-cores larger chunks, all cores
/// finish closer to the same time, reducing tail latency.
///
/// Set to 0 to disable asymmetric partitioning (pure even split).
const N_FAST_CORES: usize = 4;

/// Weight ratio for asymmetric partitioning, expressed as integers to
/// avoid floating-point on the hot path. P-cores get FAST_WEIGHT units
/// of work per thread, E-cores get SLOW_WEIGHT units. The ratio 3:2
/// gives P-cores 1.5x the rows of E-cores, matching the approximate
/// M4 P-core/E-core throughput ratio for NEON matmul.
const FAST_WEIGHT: usize = 3;
const SLOW_WEIGHT: usize = 2;

/// Work descriptor broadcast to all workers on each `parallel_for` call.
/// Contains the closure as a type-erased function pointer plus the row range.
struct WorkItem {
    /// Type-erased pointer to the closure: `&dyn Fn(usize, usize)`.
    /// Transmuted to usize to cross the Mutex boundary.
    /// SAFETY: Only valid during the `parallel_for` call that set it.
    /// Workers complete before `parallel_for` returns, so the closure
    /// (which lives on the caller's stack) is guaranteed valid.
    fn_ptr: usize,
    /// Total rows to partition across workers.
    total_rows: usize,
    /// Monotonically increasing generation counter. Workers compare their
    /// last-seen generation against this to detect new work.
    generation: u64,
    /// Whether to use asymmetric (P-core/E-core) partitioning for this dispatch.
    /// true for matmul rows (parallel_for), false for attention heads
    /// (parallel_for_heads) where items are coarse-grained.
    asymmetric: bool,
}

/// Shared state between the caller thread and persistent worker threads.
struct SharedState {
    /// Work item protected by mutex. Workers wait on `work_ready` condvar
    /// until `generation` advances past their last-seen generation.
    work: Mutex<WorkItem>,
    /// Signaled when new work is available (generation incremented).
    work_ready: Condvar,
    /// Signaled when a worker completes its chunk (`remaining` hits 0).
    work_done: Condvar,
    /// Number of workers that haven't finished their chunk yet.
    /// Caller waits until this reaches 0 (via `work_done` condvar).
    remaining: AtomicUsize,
    /// Set to true when the pool is being dropped. Workers exit their loop.
    shutdown: AtomicBool,
    /// Atomic mirror of `WorkItem::generation`. Incremented in `parallel_for`
    /// BEFORE the mutex-protected work item is updated. Workers spin on this
    /// to detect new work without acquiring the mutex, then lock only to read
    /// the work descriptor. Also bumped on shutdown so spinning workers exit.
    generation: AtomicU64,
    /// Number of worker threads (excluding caller). Used to compute chunk ranges.
    num_workers: usize,
    /// Number of "fast" P-cores for asymmetric partitioning. Clamped to
    /// at most total_threads at construction. 0 = symmetric (even split).
    n_fast: usize,
}

/// Compute the (start, end) row range for thread `thread_id` out of
/// `total_threads`, with asymmetric weighting for heterogeneous cores.
///
/// Threads 0..n_fast are "fast" (P-cores) and get FAST_WEIGHT units of
/// work. Threads n_fast..total_threads are "slow" (E-cores) and get
/// SLOW_WEIGHT units. Within each group, remainder rows are distributed
/// to the lowest-indexed threads (standard div+mod balancing).
///
/// When n_fast == 0 or n_fast >= total_threads, falls back to symmetric
/// partitioning (identical to the original algorithm).
///
/// **Even-alignment**: All chunk boundaries are aligned to 2-row pairs.
/// This ensures the NEON 2-row matmul kernel always processes complete
/// pairs within each thread's chunk, avoiding accumulator path divergence
/// between the 2-row and 1-row tail paths. The last thread absorbs any
/// odd remainder row. Cost: at most +/-1 row per thread vs. ideal split.
///
/// Invariant: the union of all thread ranges is exactly 0..total_rows
/// with no gaps or overlaps. This is verified by the test suite.
#[inline]
fn compute_chunk_range(
    thread_id: usize,
    total_threads: usize,
    total_rows: usize,
    n_fast: usize,
) -> (usize, usize) {
    // Partition in units of 2-row pairs. Any odd remainder row is
    // absorbed by the last thread (highest thread_id = caller thread).
    let total_pairs = total_rows / 2;
    let odd_tail = total_rows % 2; // 0 or 1

    let (pair_start, pair_end) = compute_pair_range(
        thread_id, total_threads, total_pairs, n_fast,
    );

    let start = pair_start * 2;
    let mut end = pair_end * 2;

    // Last thread absorbs any odd trailing row.
    if thread_id == total_threads - 1 {
        end += odd_tail;
    }

    (start, end)
}

/// Inner pair-level partitioning. Splits `total_pairs` across threads
/// using asymmetric weights (or symmetric when n_fast == 0).
#[inline]
fn compute_pair_range(
    thread_id: usize,
    total_threads: usize,
    total_pairs: usize,
    n_fast: usize,
) -> (usize, usize) {
    // Fall back to symmetric if asymmetric is disabled or not applicable.
    if n_fast == 0 || n_fast >= total_threads || total_threads <= 1 {
        let base = total_pairs / total_threads;
        let extra = total_pairs % total_threads;
        let start = thread_id * base + thread_id.min(extra);
        let chunk = base + if thread_id < extra { 1 } else { 0 };
        return (start, start + chunk);
    }

    let n_slow = total_threads - n_fast;
    let total_weight = n_fast * FAST_WEIGHT + n_slow * SLOW_WEIGHT;

    // Total pairs allocated to the fast group (integer division, floor).
    // The slow group gets the remainder to ensure exact coverage.
    let fast_total = total_pairs * n_fast * FAST_WEIGHT / total_weight;
    let slow_total = total_pairs - fast_total;

    if thread_id < n_fast {
        // This thread is a P-core. Partition fast_total among n_fast threads.
        let base = fast_total / n_fast;
        let extra = fast_total % n_fast;
        let start = thread_id * base + thread_id.min(extra);
        let chunk = base + if thread_id < extra { 1 } else { 0 };
        (start, start + chunk)
    } else {
        // This thread is an E-core. Partition slow_total among n_slow threads.
        let slow_id = thread_id - n_fast;
        let base = slow_total / n_slow;
        let extra = slow_total % n_slow;
        let start = fast_total + slow_id * base + slow_id.min(extra);
        let chunk = base + if slow_id < extra { 1 } else { 0 };
        (start, start + chunk)
    }
}

/// Persistent thread pool for parallel matmul dispatch.
///
/// Worker threads are created once at construction and parked on a condvar.
/// `parallel_for` wakes them, they do their chunk, signal completion, and sleep.
/// The caller thread also participates (handles the last chunk) to avoid waste.
pub struct ThreadPool {
    /// Persistent worker thread handles. Joined on drop.
    workers: Vec<JoinHandle<()>>,
    /// Shared state (Arc'd for workers to hold a reference).
    shared: Arc<SharedState>,
}

impl ThreadPool {
    /// Create a new persistent thread pool with the specified number of worker threads.
    ///
    /// `num_threads` is the number of *additional* threads beyond the caller.
    /// Total parallelism = num_threads + 1 (caller also does work).
    pub fn new(num_threads: usize) -> Self {
        let num_threads = num_threads.max(1);
        let total_threads = num_threads + 1; // workers + caller
        // Clamp n_fast to total_threads (can't have more fast cores than threads).
        let n_fast = N_FAST_CORES.min(total_threads);

        let shared = Arc::new(SharedState {
            work: Mutex::new(WorkItem {
                fn_ptr: 0,
                total_rows: 0,
                generation: 0,
                asymmetric: false,
            }),
            work_ready: Condvar::new(),
            work_done: Condvar::new(),
            remaining: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            num_workers: num_threads,
            n_fast,
        });

        let mut workers = Vec::with_capacity(num_threads);
        for worker_id in 0..num_threads {
            let state = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("lumen-pool-{worker_id}"))
                .spawn(move || {
                    worker_loop(state, worker_id);
                })
                .expect("failed to spawn thread pool worker");
            workers.push(handle);
        }

        Self { workers, shared }
    }

    /// Create a thread pool using available parallelism minus 1.
    /// Falls back to 1 thread if detection fails.
    pub fn with_default_threads() -> Self {
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Use all cores: caller thread + (available - 1) worker threads = available total
        let num_threads = if available > 1 { available - 1 } else { 1 };
        Self::new(num_threads)
    }

    /// Returns the total thread count (workers + caller).
    #[inline]
    pub fn total_threads(&self) -> usize {
        self.shared.num_workers + 1
    }

    /// Returns the number of worker threads (excluding the caller).
    #[inline]
    pub fn num_workers(&self) -> usize {
        self.shared.num_workers
    }

    /// Returns true if the given out_dim should be parallelized.
    ///
    /// Criteria:
    /// - out_dim >= PARALLEL_THRESHOLD (256)
    /// - Each thread gets at least MIN_ROWS_PER_THREAD (32) rows
    #[inline]
    pub fn should_parallelize(&self, out_dim: usize) -> bool {
        if out_dim < PARALLEL_THRESHOLD {
            return false;
        }
        let total_threads = self.total_threads();
        let rows_per_thread = out_dim / total_threads;
        rows_per_thread >= MIN_ROWS_PER_THREAD
    }

    /// Execute `f(start_row, end_row)` in parallel across threads.
    ///
    /// Splits `0..total_rows` into contiguous chunks: one per worker thread,
    /// plus one for the caller thread (last chunk). The caller thread participates
    /// to avoid wasting a core.
    ///
    /// SAFETY: The closure `f` is transmitted to workers as a type-erased pointer.
    /// This is safe because:
    /// 1. Workers complete their chunk before this function returns (barrier via
    ///    `remaining` counter + `work_done` condvar).
    /// 2. The closure lives on the caller's stack frame, which outlives the
    ///    workers' access to it.
    /// 3. `f: Fn + Sync` ensures concurrent read access is safe.
    ///
    /// # Panics
    ///
    /// Does not propagate worker panics (workers catch panics silently to avoid
    /// poisoning the pool). If a worker panics, `remaining` still decrements,
    /// so the caller does not deadlock.
    #[inline]
    pub fn parallel_for<F>(&self, total_rows: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        let total_threads = self.total_threads();

        // Avoid overhead for trivially small work
        if total_rows == 0 {
            return;
        }

        if total_threads <= 1 || total_rows < PARALLEL_THRESHOLD {
            f(0, total_rows);
            return;
        }

        // Type-erase the closure to a usize for transmission through the Mutex.
        // SAFETY: We transmit `&dyn Fn(usize, usize)` as a usize. The trait object
        // pointer is two words (data + vtable), so we transmit &(data, vtable) as a
        // single usize pointer to the fat pointer on the stack.
        let fn_trait_obj: &dyn Fn(usize, usize) = &f;
        let fn_ptr: usize = &fn_trait_obj as *const &dyn Fn(usize, usize) as usize;

        // Set remaining count BEFORE publishing work, so workers can't
        // decrement before we set it.
        let num_workers = self.shared.num_workers;
        self.shared.remaining.store(num_workers, Ordering::Release);

        // Publish work under the lock. The atomic generation is incremented
        // inside the lock to guarantee that when a spinning worker sees the
        // new generation and acquires the mutex, the work descriptor (fn_ptr,
        // total_rows) is already written.
        {
            let mut work = self.shared.work.lock().unwrap();
            work.fn_ptr = fn_ptr;
            work.total_rows = total_rows;
            work.asymmetric = true;
            work.generation += 1;
            // Atomic mirror: Release pairs with workers' Acquire load.
            self.shared.generation.store(work.generation, Ordering::Release);
        }
        // Wake workers that fell through to condvar sleep.
        self.shared.work_ready.notify_all();

        // Caller thread does the last chunk while workers handle the first N chunks.
        // Workers are thread_ids 0..num_workers, caller is thread_id num_workers.
        let n_fast = self.shared.n_fast;
        let (caller_start, caller_end) = compute_chunk_range(
            num_workers, total_threads, total_rows, n_fast,
        );
        if caller_start < caller_end {
            f(caller_start, caller_end);
        }

        // Wait for all workers to complete.
        // Workers decrement `remaining` and notify `work_done`.
        // Spin first: the caller just finished its own chunk, so workers are
        // likely already done or finishing. This avoids a mutex+condvar round
        // trip in the common case.
        let mut spins = 0u32;
        while self.shared.remaining.load(Ordering::Acquire) != 0 {
            if spins < SPIN_LIMIT {
                std::hint::spin_loop();
                spins += 1;
            } else {
                // Fall back to condvar to avoid wasting CPU.
                let guard = self.shared.work.lock().unwrap();
                let _guard = self.shared.work_done.wait_while(guard, |_| {
                    self.shared.remaining.load(Ordering::Acquire) != 0
                }).unwrap();
                break;
            }
        }
    }

    /// Execute `f(start_item, end_item)` in parallel, without the PARALLEL_THRESHOLD
    /// guard that `parallel_for` uses for matmul rows.
    ///
    /// Designed for parallelizing a small number of expensive work items (e.g., 16
    /// attention heads where each head does O(seq_len * head_dim) work). The caller
    /// is responsible for ensuring parallelization is worthwhile.
    ///
    /// Only parallelizes if total_threads > 1 and total_items >= total_threads
    /// (otherwise some threads would have no work). Each thread gets at least 1 item.
    ///
    /// Same safety guarantees as `parallel_for`: closure must be `Fn + Sync`,
    /// workers complete before this function returns, closure borrows are valid.
    #[inline]
    pub fn parallel_for_heads<F>(&self, total_items: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        let total_threads = self.total_threads();

        if total_items == 0 {
            return;
        }

        // Fall back to serial if only one thread or fewer items than threads
        // (to avoid threads with zero work, which still pay wake/sleep cost).
        if total_threads <= 1 || total_items < total_threads {
            f(0, total_items);
            return;
        }

        // Same dispatch mechanism as parallel_for, just without the threshold check.
        let fn_trait_obj: &dyn Fn(usize, usize) = &f;
        let fn_ptr: usize = &fn_trait_obj as *const &dyn Fn(usize, usize) as usize;

        let num_workers = self.shared.num_workers;
        self.shared.remaining.store(num_workers, Ordering::Release);

        {
            let mut work = self.shared.work.lock().unwrap();
            work.fn_ptr = fn_ptr;
            work.total_rows = total_items;
            work.asymmetric = false;
            work.generation += 1;
            self.shared.generation.store(work.generation, Ordering::Release);
        }
        self.shared.work_ready.notify_all();

        // Caller handles the last chunk (thread_id = num_workers).
        // parallel_for_heads uses symmetric partitioning (n_fast=0) because
        // attention heads are coarse-grained work items (typically 16-32),
        // and asymmetric splitting of discrete heads provides negligible benefit.
        let (caller_start, caller_end) = compute_chunk_range(
            num_workers, total_threads, total_items, 0,
        );
        if caller_start < caller_end {
            f(caller_start, caller_end);
        }

        // Spin-then-park wait for workers.
        let mut spins = 0u32;
        while self.shared.remaining.load(Ordering::Acquire) != 0 {
            if spins < SPIN_LIMIT {
                std::hint::spin_loop();
                spins += 1;
            } else {
                let guard = self.shared.work.lock().unwrap();
                let _guard = self.shared.work_done.wait_while(guard, |_| {
                    self.shared.remaining.load(Ordering::Acquire) != 0
                }).unwrap();
                break;
            }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Signal shutdown and wake all workers.
        self.shared.shutdown.store(true, Ordering::Release);
        // Bump the atomic generation so spinning workers break out of spin loop.
        self.shared.generation.fetch_add(1, Ordering::Release);
        // Wake workers that fell through to condvar sleep.
        self.shared.work_ready.notify_all();

        // Join all worker threads.
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

// SAFETY: ThreadPool is Send + Sync. The shared state is behind Arc<SharedState>
// with all fields using proper synchronization (Mutex, Condvar, atomics).
// Worker handles are Send. The pool itself does not contain any unsynchronized data.
unsafe impl Send for ThreadPool {}
unsafe impl Sync for ThreadPool {}

/// Worker thread loop. Runs until shutdown is signaled.
///
/// Each iteration:
/// 1. Spin on atomic generation counter (no lock needed).
/// 2. If spin limit exceeded, fall back to condvar wait.
/// 3. Read the work item under lock (brief critical section).
/// 4. Compute this worker's chunk range.
/// 5. Call the closure with (start, end).
/// 6. Decrement `remaining`. If last worker, notify `work_done`.
/// 7. Go back to spinning.
fn worker_loop(state: Arc<SharedState>, worker_id: usize) {
    let mut last_generation: u64 = 0;

    loop {
        // Phase 1: Spin on atomic generation counter. During active inference,
        // parallel_for fires every ~100-200 us. The spin loop catches most
        // dispatches without touching the mutex, saving ~2-5 us per wakeup.
        let mut spins = 0u32;
        loop {
            if state.shutdown.load(Ordering::Acquire) {
                return;
            }
            if state.generation.load(Ordering::Acquire) > last_generation {
                break;
            }
            std::hint::spin_loop();
            spins += 1;
            if spins >= SPIN_LIMIT {
                // Phase 2: Fall back to condvar (saves CPU when idle).
                let mut work = state.work.lock().unwrap();
                loop {
                    if state.shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    if work.generation > last_generation {
                        break;
                    }
                    work = state.work_ready.wait(work).unwrap();
                }
                break;
            }
        }

        // Phase 3: Read work descriptor under lock (brief critical section).
        // Whether we arrived via spin or condvar, we must read fn_ptr and
        // total_rows under the lock to ensure we see the complete descriptor.
        let work_snapshot = {
            let work = state.work.lock().unwrap();
            last_generation = work.generation;
            (work.fn_ptr, work.total_rows, work.asymmetric)
        };

        let (fn_ptr, total_rows, asymmetric) = work_snapshot;

        // Compute this worker's chunk range. Use asymmetric partitioning
        // for matmul rows (parallel_for), symmetric for heads (parallel_for_heads).
        let total_threads = state.num_workers + 1; // workers + caller
        let n_fast = if asymmetric { state.n_fast } else { 0 };
        let (start, end) = compute_chunk_range(
            worker_id, total_threads, total_rows, n_fast,
        );

        // Reconstruct the closure from the type-erased pointer and call it.
        // SAFETY: fn_ptr is a pointer to `&dyn Fn(usize, usize)` on the caller's stack.
        // The caller is blocked waiting for `remaining` to reach 0, so the stack frame
        // (and thus the closure and all its captures) is alive for the duration of our access.
        // Multiple workers read through the same fat pointer concurrently, which is safe
        // because F: Sync.
        if start < end {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let fn_ref: &dyn Fn(usize, usize) = unsafe {
                    *(fn_ptr as *const &dyn Fn(usize, usize))
                };
                fn_ref(start, end);
            }));
            if result.is_err() {
                // Worker panicked. We still decrement remaining so the caller
                // doesn't deadlock. The panic payload is lost.
                eprintln!("lumen-pool-{worker_id}: worker panicked during parallel_for");
            }
        }

        // Signal completion.
        if state.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            // We were the last worker. Wake the caller.
            // We need to acquire the lock briefly so the condvar notification
            // is not lost if the caller hasn't entered wait yet.
            let _guard = state.work.lock().unwrap();
            state.work_done.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new(4);
        assert_eq!(pool.num_workers(), 4);
        assert_eq!(pool.total_threads(), 5);
    }

    #[test]
    fn test_thread_pool_default() {
        let pool = ThreadPool::with_default_threads();
        assert!(pool.total_threads() >= 2);
    }

    #[test]
    fn test_thread_pool_minimum_one_thread() {
        let pool = ThreadPool::new(0);
        // Clamped to at least 1
        assert_eq!(pool.num_workers(), 1);
        assert_eq!(pool.total_threads(), 2);
    }

    #[test]
    fn test_should_parallelize_small() {
        let pool = ThreadPool::new(3);
        // 100 rows < PARALLEL_THRESHOLD(256), should NOT parallelize
        assert!(!pool.should_parallelize(100));
        // 255 rows < 256, should NOT parallelize
        assert!(!pool.should_parallelize(255));
    }

    #[test]
    fn test_should_parallelize_large() {
        let pool = ThreadPool::new(3);
        // 1024 rows / 4 total threads = 256 rows/thread >= 32, should parallelize
        assert!(pool.should_parallelize(1024));
        // 2048 rows / 4 total threads = 512 rows/thread >= 32, should parallelize
        assert!(pool.should_parallelize(2048));
    }

    #[test]
    fn test_should_parallelize_too_many_threads() {
        let pool = ThreadPool::new(100);
        // 256 rows / 101 total threads = 2 rows/thread < 32, should NOT parallelize
        assert!(!pool.should_parallelize(256));
    }

    #[test]
    fn test_parallel_for_basic() {
        let pool = ThreadPool::new(3);
        let counter = AtomicUsize::new(0);

        pool.parallel_for(1000, |start, end| {
            counter.fetch_add(end - start, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_parallel_for_covers_all_rows() {
        let pool = ThreadPool::new(3);
        let total = 1000;
        let visited: Vec<AtomicUsize> = (0..total).map(|_| AtomicUsize::new(0)).collect();

        pool.parallel_for(total, |start, end| {
            for i in start..end {
                visited[i].fetch_add(1, Ordering::Relaxed);
            }
        });

        // Every row must be visited exactly once
        for (i, v) in visited.iter().enumerate() {
            assert_eq!(
                v.load(Ordering::Relaxed),
                1,
                "row {i} visited {} times",
                v.load(Ordering::Relaxed)
            );
        }
    }

    #[test]
    fn test_parallel_for_non_divisible() {
        // 1001 rows across 4 threads (3 workers + caller)
        let pool = ThreadPool::new(3);
        let total = 1001;
        let visited: Vec<AtomicUsize> = (0..total).map(|_| AtomicUsize::new(0)).collect();

        pool.parallel_for(total, |start, end| {
            for i in start..end {
                visited[i].fetch_add(1, Ordering::Relaxed);
            }
        });

        for (i, v) in visited.iter().enumerate() {
            assert_eq!(
                v.load(Ordering::Relaxed),
                1,
                "row {i} visited {} times",
                v.load(Ordering::Relaxed)
            );
        }
    }

    #[test]
    fn test_parallel_for_zero_rows() {
        let pool = ThreadPool::new(3);
        let counter = AtomicUsize::new(0);

        pool.parallel_for(0, |_start, _end| {
            counter.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_parallel_for_single_row() {
        let pool = ThreadPool::new(3);
        let counter = AtomicUsize::new(0);

        // Single row: below threshold, runs single-threaded
        pool.parallel_for(1, |start, end| {
            counter.fetch_add(end - start, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    /// Wrapper for `*mut f32` that is `Send + Sync` (for test use).
    /// SAFETY: Caller must ensure disjoint write ranges per thread.
    #[derive(Clone, Copy)]
    struct TestSyncPtr(*mut f32);
    unsafe impl Send for TestSyncPtr {}
    unsafe impl Sync for TestSyncPtr {}
    impl TestSyncPtr {
        #[inline(always)]
        fn ptr(self) -> *mut f32 { self.0 }
    }

    #[test]
    fn test_parallel_for_borrows_stack_data() {
        // Verifies that the closure can borrow stack data
        let pool = ThreadPool::new(3);
        let input_data = vec![1.0f32; 1000];
        let mut output_data = vec![0.0f32; 1000];
        let out_ptr = TestSyncPtr(output_data.as_mut_ptr());

        // This closure borrows input_data and uses a Sync pointer for output
        pool.parallel_for(1000, |start, end| {
            for i in start..end {
                // SAFETY: each thread writes to a disjoint range
                unsafe {
                    *out_ptr.ptr().add(i) = input_data[i] * 2.0;
                }
            }
        });

        for i in 0..1000 {
            assert_eq!(output_data[i], 2.0, "output[{i}] mismatch");
        }
    }

    #[test]
    fn test_parallel_for_write_to_disjoint_slices() {
        // The canonical use case: parallel matmul writing to disjoint output rows
        let pool = ThreadPool::new(3);
        let total = 512;
        let mut output = vec![0.0f32; total];
        let out_ptr = TestSyncPtr(output.as_mut_ptr());

        pool.parallel_for(total, |start, end| {
            for i in start..end {
                // SAFETY: disjoint ranges, no overlap
                unsafe {
                    *out_ptr.ptr().add(i) = i as f32;
                }
            }
        });

        for i in 0..total {
            assert_eq!(output[i], i as f32);
        }
    }

    #[test]
    fn test_parallel_for_repeated_calls() {
        // Verify workers can be reused across multiple parallel_for calls
        let pool = ThreadPool::new(3);

        for iter in 0..10 {
            let total = 1000;
            let counter = AtomicUsize::new(0);

            pool.parallel_for(total, |start, end| {
                counter.fetch_add(end - start, Ordering::Relaxed);
            });

            assert_eq!(
                counter.load(Ordering::Relaxed),
                total,
                "iteration {iter} failed"
            );
        }
    }

    #[test]
    fn test_parallel_for_drop_is_clean() {
        // Verify drop joins workers without hanging
        let pool = ThreadPool::new(4);
        pool.parallel_for(1000, |_start, _end| {});
        drop(pool); // Should not hang or panic
    }

    // --- compute_chunk_range unit tests ---

    #[test]
    fn test_chunk_range_symmetric_covers_all() {
        // Symmetric (n_fast=0): verify every row is covered exactly once
        for total_rows in [1, 7, 10, 100, 1000, 1001, 2048, 5632] {
            for total_threads in [1, 2, 3, 4, 5, 7, 10] {
                let mut covered = vec![false; total_rows];
                for tid in 0..total_threads {
                    let (start, end) = compute_chunk_range(tid, total_threads, total_rows, 0);
                    assert!(start <= end, "start > end for tid={tid}, total={total_rows}, threads={total_threads}");
                    assert!(end <= total_rows, "end > total for tid={tid}");
                    for i in start..end {
                        assert!(!covered[i], "row {i} covered twice (symmetric, total={total_rows}, threads={total_threads})");
                        covered[i] = true;
                    }
                }
                for (i, &c) in covered.iter().enumerate() {
                    assert!(c, "row {i} not covered (symmetric, total={total_rows}, threads={total_threads})");
                }
            }
        }
    }

    #[test]
    fn test_chunk_range_asymmetric_covers_all() {
        // Asymmetric: verify every row is covered exactly once
        for total_rows in [1, 7, 10, 100, 1000, 1001, 2048, 5632] {
            for total_threads in [2, 3, 4, 5, 7, 10] {
                for n_fast in [1, 2, 3, 4] {
                    if n_fast >= total_threads { continue; }
                    let mut covered = vec![false; total_rows];
                    for tid in 0..total_threads {
                        let (start, end) = compute_chunk_range(tid, total_threads, total_rows, n_fast);
                        assert!(start <= end, "start > end for tid={tid}, total={total_rows}, threads={total_threads}, n_fast={n_fast}");
                        assert!(end <= total_rows, "end > total for tid={tid}");
                        for i in start..end {
                            assert!(
                                !covered[i],
                                "row {i} covered twice (asymmetric, total={total_rows}, threads={total_threads}, n_fast={n_fast})"
                            );
                            covered[i] = true;
                        }
                    }
                    for (i, &c) in covered.iter().enumerate() {
                        assert!(
                            c,
                            "row {i} not covered (asymmetric, total={total_rows}, threads={total_threads}, n_fast={n_fast})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_chunk_range_asymmetric_fast_gets_more() {
        // With n_fast=4 and 10 total threads, fast threads should each get
        // more rows than slow threads (for sufficiently large total_rows).
        let total_rows = 2048;
        let total_threads = 10;
        let n_fast = 4;

        let (fast_start, fast_end) = compute_chunk_range(0, total_threads, total_rows, n_fast);
        let fast_chunk = fast_end - fast_start;

        let (slow_start, slow_end) = compute_chunk_range(n_fast, total_threads, total_rows, n_fast);
        let slow_chunk = slow_end - slow_start;

        assert!(
            fast_chunk > slow_chunk,
            "fast chunk ({fast_chunk}) should be larger than slow chunk ({slow_chunk})"
        );

        // Fast should be roughly 1.5x slow (FAST_WEIGHT=3, SLOW_WEIGHT=2).
        // Allow integer rounding tolerance of +/- 2.
        let ratio = fast_chunk as f64 / slow_chunk as f64;
        assert!(
            (1.2..=1.8).contains(&ratio),
            "fast/slow ratio ({ratio:.2}) should be approximately 1.5"
        );
    }

    #[test]
    fn test_chunk_range_m4_realistic() {
        // Simulate M4: 10 threads (9 workers + caller), n_fast=4
        // For TinyLlama Q+K+V matmul: out_dim=2048
        let total_rows = 2048;
        let total_threads = 10;
        let n_fast = 4;

        let mut total_covered = 0;
        for tid in 0..total_threads {
            let (start, end) = compute_chunk_range(tid, total_threads, total_rows, n_fast);
            total_covered += end - start;
        }
        assert_eq!(total_covered, total_rows);

        // For Gate+Up: out_dim=5632
        let total_rows = 5632;
        let mut total_covered = 0;
        for tid in 0..total_threads {
            let (start, end) = compute_chunk_range(tid, total_threads, total_rows, n_fast);
            total_covered += end - start;
        }
        assert_eq!(total_covered, total_rows);
    }

    #[test]
    fn test_parallel_for_asymmetric_covers_all_rows() {
        // End-to-end test: parallel_for with asymmetric partitioning
        // covers every row exactly once. Uses 9 workers to match M4 setup.
        let pool = ThreadPool::new(9);
        let total = 2048;
        let visited: Vec<AtomicUsize> = (0..total).map(|_| AtomicUsize::new(0)).collect();

        pool.parallel_for(total, |start, end| {
            for i in start..end {
                visited[i].fetch_add(1, Ordering::Relaxed);
            }
        });

        for (i, v) in visited.iter().enumerate() {
            assert_eq!(
                v.load(Ordering::Relaxed),
                1,
                "row {i} visited {} times (asymmetric parallel_for)",
                v.load(Ordering::Relaxed)
            );
        }
    }

    #[test]
    fn test_parallel_for_heads_symmetric_with_asymmetric_pool() {
        // Even with an asymmetric pool, parallel_for_heads should use
        // symmetric partitioning (all heads visited exactly once).
        let pool = ThreadPool::new(9);
        let total_heads = 32;
        let visited: Vec<AtomicUsize> = (0..total_heads).map(|_| AtomicUsize::new(0)).collect();

        pool.parallel_for_heads(total_heads, |start, end| {
            for i in start..end {
                visited[i].fetch_add(1, Ordering::Relaxed);
            }
        });

        for (i, v) in visited.iter().enumerate() {
            assert_eq!(
                v.load(Ordering::Relaxed),
                1,
                "head {i} visited {} times (parallel_for_heads should be symmetric)",
                v.load(Ordering::Relaxed)
            );
        }
    }
}
