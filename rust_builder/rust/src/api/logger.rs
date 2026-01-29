use flutter_rust_bridge::frb;
use crate::frb_generated::StreamSink;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use lazy_static::lazy_static;

lazy_static! {
    static ref DART_LOG_SINK: RwLock<Option<StreamSink<String>>> = RwLock::new(None);
}

/// Track whether the logger has been initialized to avoid double initialization errors.
static LOGGER_INITIALIZED: AtomicBool = AtomicBool::new(false);

use log::{Level, Metadata, Record};

struct CombinedLogger;

impl log::Log for CombinedLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        // Debug builds: show all logs up to Debug level
        // Release builds: show only Info and above
        #[cfg(debug_assertions)]
        {
            metadata.level() <= Level::Debug
        }
        #[cfg(not(debug_assertions))]
        {
            metadata.level() <= Level::Info
        }
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let msg = format!("[{}] {}", record.level(), record.args());
            
            // 1. Send to Dart Stream (only if sink is available)
            send_log_to_dart(&msg);
            
            // 2. Platform native logging - always provide fallback output
            #[cfg(target_os = "android")]
            {
                // Android: println fallback (captured by Flutter debug console)
                // For production logcat integration, consider android_logger crate
                println!("{}", msg);
            }
            
            #[cfg(target_os = "ios")]
            {
                // iOS: println captured by Flutter debug console
                println!("{}", msg);
            }
            
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            {
                // Desktop: standard output
                println!("{}", msg);
            }
        }
    }

    fn flush(&self) {}
}

static LOGGER: CombinedLogger = CombinedLogger;

/// Initialize the global logger.
/// 
/// This function is idempotent - calling it multiple times is safe and will
/// simply return Ok(()) if the logger is already initialized.
/// 
/// Log levels:
/// - Debug builds: DEBUG and above
/// - Release builds: INFO and above
pub fn init_logger() -> anyhow::Result<()> {
    // Check if already initialized using atomic compare-exchange
    if LOGGER_INITIALIZED.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        // Already initialized, return success silently
        return Ok(());
    }
    
    // Set appropriate log level based on build type
    #[cfg(debug_assertions)]
    let level = log::LevelFilter::Debug;
    #[cfg(not(debug_assertions))]
    let level = log::LevelFilter::Info;
    
    log::set_logger(&LOGGER)
        .map(|()| log::set_max_level(level))
        .map_err(|e| {
            // Reset the flag if initialization failed
            LOGGER_INITIALIZED.store(false, Ordering::SeqCst);
            anyhow::anyhow!("Logger init failed: {}", e)
        })
}
/// Initialize the Dart log stream.
/// Call this from Dart to start receiving Rust logs.
#[frb(sync)]
pub fn init_log_stream(sink: StreamSink<String>) -> anyhow::Result<()> {
    let mut guard = DART_LOG_SINK.write().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
    *guard = Some(sink);
    Ok(())
}

/// Close the Dart log stream.
/// Call this when disposing the log subscription to prevent memory leaks.
#[frb(sync)]
pub fn close_log_stream() -> anyhow::Result<()> {
    let mut guard = DART_LOG_SINK.write().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
    *guard = None;
    Ok(())
}

/// Helper to send a log message to Dart if the stream is active.
/// Takes a reference to avoid unnecessary cloning when sink is not available.
pub fn send_log_to_dart(msg: &str) {
    match DART_LOG_SINK.read() {
        Ok(guard) => {
            if let Some(sink) = &*guard {
                // Only clone when we actually have a sink to send to
                let _ = sink.add(msg.to_string());
            }
        }
        Err(_) => {
            // RwLock is poisoned - only warn in debug builds
            #[cfg(debug_assertions)]
            eprintln!("[WARNING] Dart log sink lock is poisoned, log message dropped");
        }
    }
}
