use log::Level;
use nccl_net_sys::{ncclDebugLogLevel, ncclDebugLogger_t};
use std::ffi::CString;

static mut LOGGER: Logger = Logger(None);

pub struct Logger(ncclDebugLogger_t);

impl Logger {
    pub fn init(
        level: log::LevelFilter,
        logger: ncclDebugLogger_t,
    ) -> Result<(), log::SetLoggerError> {
        unsafe {
            LOGGER.0 = logger;
            log::set_logger(&LOGGER).map(|()| log::set_max_level(level))
        }
    }
}

impl log::Log for Logger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        if let Some(logger) = self.0 {
            let level = match record.level() {
                Level::Error => ncclDebugLogLevel::NCCL_LOG_WARN,
                Level::Warn => ncclDebugLogLevel::NCCL_LOG_WARN,
                Level::Info => ncclDebugLogLevel::NCCL_LOG_INFO,
                Level::Debug => ncclDebugLogLevel::NCCL_LOG_TRACE,
                Level::Trace => ncclDebugLogLevel::NCCL_LOG_TRACE,
            };

            let file = record.file().unwrap_or_default();
            let file = CString::new(file).unwrap_or_default();

            let args = format!("{}", record.args());
            let args = CString::new(args).unwrap_or_default();

            unsafe {
                logger(
                    level,
                    u64::MAX,
                    file.as_ptr(),
                    record.line().unwrap_or_default().try_into().unwrap(),
                    args.as_ptr(),
                );
            }
        }
    }

    fn flush(&self) {}
}
