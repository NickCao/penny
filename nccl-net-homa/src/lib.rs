#![feature(ip)]
#![feature(cstr_from_bytes_until_nul)]
#![feature(strict_provenance)]
use crate::homa::*;
use core::slice;
use nccl_net_sys::*;
use std::ffi::{c_int, c_void};
use std::ptr::null_mut;

pub mod error;
pub mod homa;
pub mod logger;

pub extern "C" fn init(logger: ncclDebugLogger_t) -> ncclResult_t {
    logger::Logger::init(log::LevelFilter::Debug, logger).unwrap();
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn devices(ndev: *mut c_int) -> ncclResult_t {
    match homa::Homa::devices() {
        Ok(n) => {
            *ndev = n;
            ncclResult_t::ncclSuccess
        }
        Err(err) => err.into(),
    }
}

unsafe extern "C" fn get_properties(
    dev: c_int,
    props: *mut ncclNetProperties_v6_t,
) -> ncclResult_t {
    match homa::Homa::get_properties(dev) {
        Ok(p) => {
            *props = p;
            ncclResult_t::ncclSuccess
        }
        Err(err) => err.into(),
    }
}

unsafe extern "C" fn listen(
    dev: c_int,
    handle: *mut c_void,
    listen_comm: *mut *mut c_void,
) -> ncclResult_t {
    let handle = slice::from_raw_parts_mut(handle.cast(), NCCL_NET_HANDLE_MAXSIZE as usize);
    match homa::Homa::listen(dev, handle) {
        Ok(comm) => {
            *listen_comm.cast() = Box::into_raw(Box::new(comm));
            ncclResult_t::ncclSuccess
        }
        Err(err) => err.into(),
    }
}

unsafe extern "C" fn connect(
    dev: c_int,
    handle: *mut c_void,
    send_comm: *mut *mut c_void,
) -> ncclResult_t {
    let handle = slice::from_raw_parts(handle.cast(), NCCL_NET_HANDLE_MAXSIZE as usize);
    match homa::Homa::connect(dev, handle) {
        Ok(comm) => {
            *send_comm.cast() = Box::into_raw(Box::new(comm));
            ncclResult_t::ncclSuccess
        }
        Err(err) => err.into(),
    }
}

unsafe extern "C" fn accept(listen_comm: *mut c_void, recv_comm: *mut *mut c_void) -> ncclResult_t {
    let listen_comm = &mut *(listen_comm.cast());
    match homa::Homa::accept(listen_comm) {
        Ok(comm) => {
            *recv_comm.cast() = Box::into_raw(Box::new(comm));
            ncclResult_t::ncclSuccess
        }
        Err(err) => err.into(),
    }
}

extern "C" fn reg_mr(
    _comm: *mut c_void,
    _data: *mut c_void,
    _size: c_int,
    _type_: c_int,
    _mhandle: *mut *mut c_void,
) -> ncclResult_t {
    ncclResult_t::ncclSuccess
}

extern "C" fn dereg_mr(_comm: *mut c_void, _mhandle: *mut c_void) -> ncclResult_t {
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn isend(
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
    match Homa::isend(send_comm, data) {
        Ok(Some(req)) => {
            *(request.cast()) = Box::into_raw(Box::new(req));
            ncclResult_t::ncclSuccess
        }
        Ok(None) => {
            *request = null_mut();
            ncclResult_t::ncclSuccess
        }
        Err(err) => err.into(),
    }
}

unsafe extern "C" fn irecv(
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
    let buf = slice::from_raw_parts_mut(data[0].cast(), sizes[0].try_into().unwrap());
    let recv_comm = &mut *(recv_comm.cast());
    match Homa::irecv(recv_comm, buf) {
        Ok(req) => {
            *(request.cast()) = Box::into_raw(Box::new(req));
            ncclResult_t::ncclSuccess
        }
        Err(err) => err.into(),
    }
}

unsafe extern "C" fn test(
    request: *mut c_void,
    done: *mut c_int,
    sizes: *mut c_int,
) -> ncclResult_t {
    let request = &mut *(request.cast());
    match Homa::test(request) {
        Ok(Some(size)) => {
            *done = 1;
            if let Some(sizes) = sizes.as_mut() {
                *sizes = size
            }
            ncclResult_t::ncclSuccess
        }
        Ok(None) => {
            *done = 0;
            ncclResult_t::ncclSuccess
        }
        Err(err) => err.into(),
    }
}

unsafe extern "C" fn close_send(send_comm: *mut c_void) -> ncclResult_t {
    let send_comm = Box::from_raw(send_comm.cast());
    match Homa::close_send(*send_comm) {
        Ok(_) => ncclResult_t::ncclSuccess,
        Err(err) => err.into(),
    }
}

unsafe extern "C" fn close_recv(recv_comm: *mut c_void) -> ncclResult_t {
    let recv_comm = Box::from_raw(recv_comm.cast());
    match Homa::close_recv(*recv_comm) {
        Ok(_) => ncclResult_t::ncclSuccess,
        Err(err) => err.into(),
    }
}

unsafe extern "C" fn close_listen(listen_comm: *mut c_void) -> ncclResult_t {
    let listen_comm = Box::from_raw(listen_comm.cast());
    match Homa::close_listen(*listen_comm) {
        Ok(_) => ncclResult_t::ncclSuccess,
        Err(err) => err.into(),
    }
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
