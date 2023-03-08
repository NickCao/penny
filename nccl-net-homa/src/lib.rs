#![feature(ip)]

use crate::homa::*;
use core::slice;
use homa::ListenComm;
use nccl_net_sys::*;
use socket2::SockAddr;
use std::ffi::CString;
use std::ffi::{c_int, c_void};
pub mod homa;

static mut LOGGER: ncclDebugLogger_t = None;

macro_rules! log {
    ($level:expr, $sys:expr, $($arg:tt)*) => {
        unsafe {
            if let Some(logger) = LOGGER {
                let file = CString::new(file!()).unwrap();
                let fmt = CString::new(format!($($arg)*)).unwrap();
                logger(
                    $level,
                    $sys.0.try_into().unwrap(),
                    file.as_ptr(),
                    line!() as i32,
                    fmt.as_ptr(),
                );
            }
        }
    };
}

pub extern "C" fn init(logger: ncclDebugLogger_t) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        ncclDebugLogSubSys::NCCL_INIT,
        "init",
    );
    unsafe {
        LOGGER = logger;
    }

    ncclResult_t::ncclSuccess
}

pub extern "C" fn devices(ndev: *mut c_int) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        ncclDebugLogSubSys::NCCL_INIT,
        "devices",
    );
    let ndev = unsafe { ndev.as_mut().unwrap() };
    homa::Homa::devices(ndev)
}

pub extern "C" fn get_properties(dev: c_int, props: *mut ncclNetProperties_v6_t) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        ncclDebugLogSubSys::NCCL_INIT,
        "get_properties",
    );
    let props = unsafe { props.as_mut().unwrap() };
    homa::Homa::get_properties(dev, props)
}

pub unsafe extern "C" fn listen(
    dev: c_int,
    handle: *mut c_void,
    listen_comm: *mut *mut c_void,
) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        ncclDebugLogSubSys::NCCL_INIT,
        "listen",
    );
    let handle = unsafe { &mut *(handle as *mut SockAddr) };
    let (comm, result) = homa::Homa::listen(dev, handle);
    *(listen_comm as *mut *mut ListenComm) = Box::into_raw(Box::new(comm));
    result
}

pub unsafe extern "C" fn connect(
    dev: c_int,
    handle: *mut c_void,
    send_comm: *mut *mut c_void,
) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        ncclDebugLogSubSys::NCCL_INIT,
        "connect",
    );
    let handle = unsafe { &*(handle as *mut SockAddr) };
    let (comm, result) = homa::Homa::connect(dev, handle);
    *(send_comm as *mut *mut SendComm) = Box::into_raw(Box::new(comm));
    result
}

pub unsafe extern "C" fn accept(
    listen_comm: *mut c_void,
    recv_comm: *mut *mut c_void,
) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        ncclDebugLogSubSys::NCCL_INIT,
        "accept",
    );
    let listen_comm = unsafe { &mut *(listen_comm as *mut ListenComm) };
    let (comm, result) = Homa::accept(listen_comm);
    *(recv_comm as *mut *mut RecvComm) = Box::into_raw(Box::new(comm));
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
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        ncclDebugLogSubSys::NCCL_INIT,
        "isend",
    );
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
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        ncclDebugLogSubSys::NCCL_INIT,
        "irecv",
    );
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
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        ncclDebugLogSubSys::NCCL_INIT,
        "test",
    );
    let request: &mut Request = unsafe { &mut *request.cast() };
    let done = unsafe { done.as_mut().unwrap() };
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
