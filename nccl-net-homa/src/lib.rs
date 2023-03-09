#![feature(ip)]
#![feature(cstr_from_bytes_until_nul)]
#![feature(strict_provenance)]
use crate::homa::*;
use core::slice;
use nccl_net_sys::*;
use std::ffi::{c_int, c_void};
use std::ptr::null_mut;

pub mod homa;
pub mod logger;

pub extern "C" fn init(logger: ncclDebugLogger_t) -> ncclResult_t {
    logger::Logger::init(log::LevelFilter::Debug, logger).unwrap();
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn devices(ndev: *mut c_int) -> ncclResult_t {
    match homa::Homa::devices() {
        Ok(n) => {
            if let Ok(n) = n.try_into() {
                *ndev = n;
                ncclResult_t::ncclSuccess
            } else {
                ncclResult_t::ncclInternalError
            }
        }
        Err(err) => err,
    }
}

#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn get_properties(
    dev: c_int,
    props: *mut ncclNetProperties_v6_t,
) -> ncclResult_t {
    let props = &mut *props;
    homa::Homa::get_properties(dev, props)
}

#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn listen(
    dev: c_int,
    handle: *mut c_void,
    listen_comm: *mut *mut c_void,
) -> ncclResult_t {
    let handle = slice::from_raw_parts_mut(handle.cast(), NCCL_NET_HANDLE_MAXSIZE as usize);
    let (comm, result) = homa::Homa::listen(dev, handle);
    *(listen_comm as *mut *mut ListenComm) = Box::into_raw(Box::new(comm));
    result
}

#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn connect(
    dev: c_int,
    handle: *mut c_void,
    send_comm: *mut *mut c_void,
) -> ncclResult_t {
    let handle = slice::from_raw_parts(handle.cast(), NCCL_NET_HANDLE_MAXSIZE as usize);
    let (comm, result) = homa::Homa::connect(dev, handle);
    *(send_comm as *mut *mut SendComm) = Box::into_raw(Box::new(comm));
    result
}

#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn accept(
    listen_comm: *mut c_void,
    recv_comm: *mut *mut c_void,
) -> ncclResult_t {
    let listen_comm = &mut *(listen_comm.cast());
    let (comm, result) = Homa::accept(listen_comm);
    *(recv_comm as *mut *mut RecvComm) = Box::into_raw(Box::new(comm));
    result
}

#[allow(clippy::missing_safety_doc)]
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

#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn isend(
    send_comm: *mut c_void,
    data: *mut c_void,
    size: c_int,
    _tag: c_int,
    _mhandle: *mut c_void,
    request: *mut *mut c_void,
) -> ncclResult_t {
    let size: usize = size.try_into().unwrap();
    let data = slice::from_raw_parts(data.cast(), size);
    let send_comm = &mut *(send_comm.cast());
    let (req, result) = Homa::isend(send_comm, data);
    if let Some(req) = req {
        *(request as *mut *mut Request) = Box::into_raw(Box::new(req));
    } else {
        *(request as *mut *mut Request) = null_mut();
    }
    result
}

#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn irecv(
    recv_comm: *mut c_void,
    n: c_int,
    data: *mut *mut c_void,
    sizes: *mut c_int,
    _tags: *mut c_int,
    _mhandles: *mut *mut c_void,
    request: *mut *mut c_void,
) -> ncclResult_t {
    let n: usize = n.try_into().unwrap();
    let data = slice::from_raw_parts(data, n);
    let sizes = slice::from_raw_parts(sizes, n);
    let buffer = slice::from_raw_parts_mut(data[0].cast(), sizes[0].try_into().unwrap());
    let recv_comm = &mut *(recv_comm.cast());
    let (req, result) = Homa::irecv(recv_comm, buffer);
    *(request as *mut *mut Request) = Box::into_raw(Box::new(req));
    result
}

#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn test(
    request: *mut c_void,
    done: *mut c_int,
    sizes: *mut c_int,
) -> ncclResult_t {
    let request = &mut *(request.cast());
    let done = &mut *done;
    let sizes = sizes.as_mut(); // FIXME: sizes is an array
    Homa::test(request, done, sizes)
}

unsafe extern "C" fn close_send(send_comm: *mut c_void) -> ncclResult_t {
    let send_comm = Box::from_raw(send_comm.cast());
    Homa::close_send(*send_comm)
}

unsafe extern "C" fn close_recv(recv_comm: *mut c_void) -> ncclResult_t {
    let recv_comm = Box::from_raw(recv_comm.cast());
    Homa::close_recv(*recv_comm)
}

unsafe extern "C" fn close_listen(listen_comm: *mut c_void) -> ncclResult_t {
    let listen_comm = Box::from_raw(listen_comm.cast());
    Homa::close_listen(*listen_comm)
}

#[export_name = "ncclNetPlugin_v6"]
static mut PLUGIN: ncclNet_v6_t = ncclNet_v6_t {
    name: b"homa\0".as_ptr().cast(),
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
