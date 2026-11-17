pub(crate) fn parallel_for(tasks: usize, body: &(dyn Fn(usize) + Sync)) {
    #[cfg(not(target_arch = "wasm32"))]
    if tasks > 1 {
        native::pool().run(tasks, body);
        return;
    }
    for task in 0..tasks {
        body(task);
    }
}

pub(crate) fn threads() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        native::pool().threads()
    }
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
}

pub(crate) struct SharedPtr<T>(pub *mut T);

unsafe impl<T> Send for SharedPtr<T> {}
unsafe impl<T> Sync for SharedPtr<T> {}

impl<T> SharedPtr<T> {
    pub fn get(&self) -> *mut T {
        self.0
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::cell::UnsafeCell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    const MAX_THREADS: usize = 16;
    const DEFAULT_THREADS: usize = 8;
    const SPIN_BEFORE_SLEEP: Duration = Duration::from_micros(1500);

    type Body<'a> = dyn Fn(usize) + Sync + 'a;

    struct Shared {
        state: AtomicUsize,
        active: AtomicUsize,
        cursors: [Cursor; MAX_THREADS],
        segments: AtomicUsize,
        body: UnsafeCell<Option<*const Body<'static>>>,
        panicked: AtomicBool,
        sleepers: AtomicUsize,
        sleep_lock: Mutex<()>,
        wake: Condvar,
    }

    #[repr(align(64))]
    struct Cursor(AtomicU64);

    impl Cursor {
        fn set(&self, front: usize, back: usize) {
            self.0
                .store((front as u64) << 32 | back as u64, Ordering::Relaxed);
        }

        fn take(&self, from_front: bool) -> Option<usize> {
            let mut current = self.0.load(Ordering::Relaxed);
            loop {
                let (front, back) = (current >> 32, current & 0xffff_ffff);
                if front >= back {
                    return None;
                }
                let (next, task) = if from_front {
                    ((front + 1) << 32 | back, front)
                } else {
                    (front << 32 | (back - 1), back - 1)
                };
                match self.0.compare_exchange_weak(
                    current,
                    next,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Some(task as usize),
                    Err(actual) => current = actual,
                }
            }
        }
    }

    impl Shared {
        fn claim(&self, home: usize, segments: usize) -> Option<usize> {
            if let Some(task) = self.cursors[home].take(true) {
                return Some(task);
            }
            (1..segments).find_map(|step| self.cursors[(home + step) % segments].take(false))
        }
    }

    unsafe impl Sync for Shared {}
    unsafe impl Send for Shared {}

    pub(super) struct Pool {
        shared: Arc<Shared>,
        workers: usize,
        run_lock: Mutex<()>,
    }

    pub(super) fn pool() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        POOL.get_or_init(|| Pool::new(thread_count()))
    }

    fn thread_count() -> usize {
        if let Some(count) = std::env::var("NNCOSS_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|count| *count > 0)
        {
            return count.min(MAX_THREADS);
        }
        let logical = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1);
        let physical = if cfg!(target_arch = "x86_64") {
            logical.div_ceil(2)
        } else {
            logical
        };
        physical.clamp(1, DEFAULT_THREADS)
    }

    impl Pool {
        fn new(threads: usize) -> Self {
            let shared = Arc::new(Shared {
                state: AtomicUsize::new(0),
                active: AtomicUsize::new(0),
                cursors: std::array::from_fn(|_| Cursor(AtomicU64::new(0))),
                segments: AtomicUsize::new(1),
                body: UnsafeCell::new(None),
                panicked: AtomicBool::new(false),
                sleepers: AtomicUsize::new(0),
                sleep_lock: Mutex::new(()),
                wake: Condvar::new(),
            });
            let mut workers = 0;
            for index in 1..threads.min(MAX_THREADS) {
                let shared = Arc::clone(&shared);
                let spawned = std::thread::Builder::new()
                    .name(format!("nncoss-worker-{index}"))
                    .spawn(move || worker(&shared, index));
                if spawned.is_ok() {
                    workers += 1;
                }
            }
            Self {
                shared,
                workers,
                run_lock: Mutex::new(()),
            }
        }

        pub(super) fn threads(&self) -> usize {
            self.workers + 1
        }

        pub(super) fn run(&self, tasks: usize, body: &Body<'_>) {
            let guard = match self.run_lock.try_lock() {
                Ok(guard) if self.workers > 0 && tasks <= u32::MAX as usize => guard,
                _ => {
                    for task in 0..tasks {
                        body(task);
                    }
                    return;
                }
            };
            let shared = &*self.shared;
            let erased: *const Body<'static> =
                unsafe { std::mem::transmute::<&Body<'_>, &'static Body<'static>>(body) };
            unsafe { *shared.body.get() = Some(erased) };
            let segments = (self.workers + 1).min(tasks);
            shared.segments.store(segments, Ordering::Relaxed);
            for (segment, cursor) in shared.cursors[..segments].iter().enumerate() {
                cursor.set(tasks * segment / segments, tasks * (segment + 1) / segments);
            }
            shared.panicked.store(false, Ordering::Relaxed);
            shared.state.fetch_add(1, Ordering::SeqCst);
            if shared.sleepers.load(Ordering::SeqCst) > 0 {
                let _sleep = shared
                    .sleep_lock
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                shared.wake.notify_all();
            }
            let close = Close(shared);
            while let Some(task) = shared.claim(0, segments) {
                body(task);
            }
            drop(close);
            drop(guard);
            if shared.panicked.load(Ordering::Relaxed) {
                panic!("并行任务在工作线程中 panic");
            }
        }
    }

    struct Close<'a>(&'a Shared);

    impl Drop for Close<'_> {
        fn drop(&mut self) {
            self.0.state.fetch_add(1, Ordering::SeqCst);
            while self.0.active.load(Ordering::SeqCst) != 0 {
                std::hint::spin_loop();
            }
        }
    }

    fn open_state(shared: &Shared, seen: usize, ordering: Ordering) -> bool {
        let state = shared.state.load(ordering);
        state != seen && state & 1 == 1
    }

    fn wait_for_job(shared: &Shared, seen: usize) {
        let mut spins = 0_u32;
        let mut started = None;
        loop {
            if open_state(shared, seen, Ordering::Acquire) {
                return;
            }
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins % 512 != 0 {
                continue;
            }
            let now = Instant::now();
            let since = *started.get_or_insert(now);
            if now.duration_since(since) < SPIN_BEFORE_SLEEP {
                continue;
            }
            let mut guard = shared
                .sleep_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            shared.sleepers.fetch_add(1, Ordering::SeqCst);
            while !open_state(shared, seen, Ordering::SeqCst) {
                guard = shared
                    .wake
                    .wait(guard)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            shared.sleepers.fetch_sub(1, Ordering::SeqCst);
            return;
        }
    }

    fn worker(shared: &Shared, index: usize) {
        let mut seen = 0;
        loop {
            wait_for_job(shared, seen);
            shared.active.fetch_add(1, Ordering::SeqCst);
            let state = shared.state.load(Ordering::SeqCst);
            if state != seen && state & 1 == 1 {
                seen = state;
                let segments = shared.segments.load(Ordering::Relaxed);
                if let Some(body) = unsafe { *shared.body.get() } {
                    let body = unsafe { &*body };
                    while let Some(task) = shared.claim(index % segments, segments) {
                        if catch_unwind(AssertUnwindSafe(|| body(task))).is_err() {
                            shared.panicked.store(true, Ordering::Relaxed);
                        }
                    }
                }
            }
            shared.active.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_task_runs_exactly_once() {
        for tasks in [0, 1, 2, 7, 64, 1000] {
            let counters: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
            for _ in 0..50 {
                parallel_for(tasks, &|task| {
                    counters[task].fetch_add(1, Ordering::Relaxed);
                });
            }
            assert!(
                counters
                    .iter()
                    .all(|counter| counter.load(Ordering::Relaxed) == 50)
            );
        }
    }

    #[test]
    fn nested_and_concurrent_calls_complete() {
        let total = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..20 {
                        parallel_for(8, &|_| {
                            parallel_for(4, &|_| {
                                total.fetch_add(1, Ordering::Relaxed);
                            });
                        });
                    }
                });
            }
        });
        assert_eq!(total.load(Ordering::Relaxed), 4 * 20 * 8 * 4);
    }

    #[test]
    fn disjoint_writes_through_shared_pointer() {
        let mut values = vec![0_usize; 4096];
        let pointer = SharedPtr(values.as_mut_ptr());
        parallel_for(64, &|task| {
            for offset in 0..64 {
                unsafe { pointer.get().add(task * 64 + offset).write(task) };
            }
        });
        for (index, value) in values.iter().enumerate() {
            assert_eq!(*value, index / 64);
        }
    }
}
