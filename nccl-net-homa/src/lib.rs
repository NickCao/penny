#![feature(ip)]
#![feature(cstr_from_bytes_until_nul)]

use crate::homa::*;
use core::slice;
use homa::ListenComm;
use log::Level;
use nccl_net_sys::*;

use std::ffi::CString;
use std::ffi::{c_int, c_void};
mod homa;

static mut LOGGER: Logger = Logger(None);

struct Logger(ncclDebugLogger_t);

impl Logger {
    fn init(level: log::LevelFilter, logger: ncclDebugLogger_t) -> Result<(), log::SetLoggerError> {
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

pub extern "C" fn init(logger: ncclDebugLogger_t) -> ncclResult_t {
    Logger::init(log::LevelFilter::Debug, logger).unwrap();

    log::debug!("init");

    ncclResult_t::ncclSuccess
}

pub extern "C" fn devices(ndev: *mut c_int) -> ncclResult_t {
    log::debug!("devices");
    let ndev = unsafe { &mut *ndev };
    homa::Homa::devices(ndev)
}

pub extern "C" fn get_properties(dev: c_int, props: *mut ncclNetProperties_v6_t) -> ncclResult_t {
    log::debug!("get_properties");
    let props = unsafe { &mut *props };
    homa::Homa::get_properties(dev, props)
}

pub extern "C" fn listen(
    dev: c_int,
    handle: *mut c_void,
    listen_comm: *mut *mut c_void,
) -> ncclResult_t {
    log::debug!("listen");
    let handle =
        unsafe { slice::from_raw_parts_mut(handle as *mut u8, NCCL_NET_HANDLE_MAXSIZE as usize) };
    let (comm, result) = homa::Homa::listen(dev, handle);
    unsafe { *(listen_comm as *mut *mut ListenComm) = Box::into_raw(Box::new(comm)) };
    result
}

pub extern "C" fn connect(
    dev: c_int,
    handle: *mut c_void,
    send_comm: *mut *mut c_void,
) -> ncclResult_t {
    log::debug!("connect");
    let handle =
        unsafe { slice::from_raw_parts(handle as *const u8, NCCL_NET_HANDLE_MAXSIZE as usize) };
    let (comm, result) = homa::Homa::connect(dev, handle);
    unsafe { *(send_comm as *mut *mut SendComm) = Box::into_raw(Box::new(comm)) };
    result
}

pub extern "C" fn accept(listen_comm: *mut c_void, recv_comm: *mut *mut c_void) -> ncclResult_t {
    log::debug!("accept");
    let listen_comm = unsafe { &mut *(listen_comm as *mut ListenComm) };
    let (comm, result) = Homa::accept(listen_comm);
    unsafe { *(recv_comm as *mut *mut RecvComm) = Box::into_raw(Box::new(comm)) };
    result
}

extern "C" fn reg_mr(
    comm: *mut c_void,
    data: *mut c_void,
    size: c_int,
    type_: c_int,
    mhandle: *mut *mut c_void,
) -> ncclResult_t {
    Homa::reg_mr(comm, data, size, type_, mhandle)
}

extern "C" fn dereg_mr(comm: *mut c_void, mhandle: *mut c_void) -> ncclResult_t {
    Homa::dereg_mr(comm, mhandle)
}

pub extern "C" fn isend(
    send_comm: *mut c_void,
    data: *mut c_void,
    size: c_int,
    _tag: c_int,
    _mhandle: *mut c_void,
    request: *mut *mut c_void,
) -> ncclResult_t {
    log::debug!("isend");
    let size: usize = size.try_into().unwrap();
    let data = unsafe { slice::from_raw_parts(data.cast(), size) };
    let send_comm = unsafe { &mut *(send_comm as *mut SendComm) };
    let (req, result) = Homa::isend(send_comm, data);
    unsafe { *(request as *mut *mut Request) = Box::into_raw(Box::new(req)) };
    result
}

pub extern "C" fn irecv(
    recv_comm: *mut c_void,
    n: c_int,
    data: *mut *mut c_void,
    sizes: *mut c_int,
    _tags: *mut c_int,
    _mhandles: *mut *mut c_void,
    request: *mut *mut c_void,
) -> ncclResult_t {
    log::debug!("irecv");
    let n: usize = n.try_into().unwrap();
    let data = unsafe { slice::from_raw_parts(data, n) };
    let sizes = unsafe { slice::from_raw_parts(sizes, n) };
    let buffer = unsafe { slice::from_raw_parts_mut(data[0].cast(), sizes[0].try_into().unwrap()) };
    let recv_comm = unsafe { &mut *(recv_comm as *mut RecvComm) };
    let (req, result) = Homa::irecv(recv_comm, buffer);
    unsafe { *(request as *mut *mut Request) = Box::into_raw(Box::new(req)) };
    result
}

pub extern "C" fn test(request: *mut c_void, done: *mut c_int, sizes: *mut c_int) -> ncclResult_t {
    log::debug!("test");
    let request = unsafe { &mut *(request as *mut Request) };
    let done = unsafe { &mut *done };
    let sizes = unsafe { sizes.as_mut() }; // FIXME: sizes is an array
    Homa::test(request, done, sizes)
}

pub extern "C" fn close_send(send_comm: *mut c_void) -> ncclResult_t {
    let comm = unsafe { Box::from_raw(send_comm as *mut SendComm) };
    Homa::close_send(*comm)
}

pub extern "C" fn close_recv(recv_comm: *mut c_void) -> ncclResult_t {
    let comm = unsafe { Box::from_raw(recv_comm as *mut RecvComm) };
    Homa::close_recv(*comm)
}

pub extern "C" fn close_listen(listen_comm: *mut c_void) -> ncclResult_t {
    let comm = unsafe { Box::from_raw(listen_comm as *mut ListenComm) };
    Homa::close_listen(*comm)
}

#[no_mangle]
pub static mut ncclNetPlugin_v6: ncclNet_v6_t = ncclNet_v6_t {
    name: b"example\0".as_ptr().cast(),
    init: Some(init),
    devices: Some(devices),
    getProperties: Some(get_properties),
    listen: Some(listen),
    connect: Some(connect),
    accept: Some(accept),
    regMr: Some(reg_mr),
    regMrDmaBuf: None,
    deregMr: Some(dereg_mr),
    isend: Some(isend),
    irecv: Some(irecv),
    iflush: None,
    test: Some(test),
    closeSend: Some(close_send),
    closeRecv: Some(close_recv),
    closeListen: Some(close_listen),
};
