#![feature(ip)]
#![feature(cstr_from_bytes_until_nul)]
#![feature(strict_provenance)]
#![feature(c_variadic)]

use nccl_net_sys::ncclNet_v6_t;

mod binding;
pub mod error;
pub mod homa;
pub mod logger;

#[export_name = "ncclNetPlugin_v6"]
pub static mut PLUGIN: ncclNet_v6_t = ncclNet_v6_t {
    name: b"homa\0".as_ptr().cast(),
    init: Some(binding::init),
    devices: Some(binding::devices),
    getProperties: Some(binding::get_properties),
    listen: Some(binding::listen),
    connect: Some(binding::connect),
    accept: Some(binding::accept),
    regMr: Some(binding::reg_mr),
    regMrDmaBuf: None,
    deregMr: Some(binding::dereg_mr),
    isend: Some(binding::isend),
    irecv: Some(binding::irecv),
    iflush: None,
    test: Some(binding::test),
    closeSend: Some(binding::close_send),
    closeRecv: Some(binding::close_recv),
    closeListen: Some(binding::close_listen),
};

#[cfg(test)]
mod test {
    use crate::binding::*;
    use nccl_net_sys::{ncclDebugLogLevel, ncclResult_t, NCCL_NET_HANDLE_MAXSIZE};
    use std::{
        ffi::{c_char, c_int, c_ulong, c_void, CStr},
        ptr::null_mut,
    };

    unsafe extern "C" fn logger(
        level: ncclDebugLogLevel,
        flags: c_ulong,
        file: *const c_char,
        line: c_int,
        fmt: *const c_char,
        ...
    ) {
        println!(
            "{:?} {} {} {} {}",
            level,
            flags,
            CStr::from_ptr(file).to_str().unwrap(),
            line,
            CStr::from_ptr(fmt).to_str().unwrap()
        )
    }

    #[test]
    fn roundtrip() {
        unsafe {
            let ret = init(Some(logger));
            assert_eq!(ret, ncclResult_t::ncclSuccess);

            let mut handle = [0u8; NCCL_NET_HANDLE_MAXSIZE as usize];
            let mut listen_comm: *mut c_void = null_mut();
            let ret = listen(0, handle.as_mut_ptr().cast(), &mut listen_comm);
            assert_eq!(ret, ncclResult_t::ncclSuccess);

            let mut send_comm: *mut c_void = null_mut();
            let ret = connect(0, handle.as_mut_ptr().cast(), &mut send_comm);
            assert_eq!(ret, ncclResult_t::ncclSuccess);

            let mut recv_comm: *mut c_void = null_mut();
            let ret = accept(listen_comm, &mut recv_comm);
            assert_eq!(ret, ncclResult_t::ncclSuccess);

            let mut data = b"hello penny\0".to_vec();
            let mut send_req: *mut c_void = null_mut();
            let ret = isend(
                send_comm,
                data.as_mut_ptr().cast(),
                data.len().try_into().unwrap(),
                0,
                null_mut(),
                &mut send_req,
            );
            assert_eq!(ret, ncclResult_t::ncclSuccess);

            let mut recv_req: *mut c_void = null_mut();
            let mut buf = vec![0u8; 100];
            let mut bufs = [buf.as_mut_ptr().cast()];
            let mut sizes = [data.len().try_into().unwrap()];
            let ret = irecv(
                recv_comm,
                1,
                bufs.as_mut_ptr(),
                sizes.as_mut_ptr(),
                null_mut(),
                null_mut(),
                &mut recv_req,
            );
            assert_eq!(ret, ncclResult_t::ncclSuccess);

            loop {
                let mut done = 0;
                let mut size = 0;
                let ret = test(recv_req, &mut done, &mut size);
                assert_eq!(ret, ncclResult_t::ncclSuccess);
                if done == 1 {
                    assert_eq!(size, data.len().try_into().unwrap());
                    break;
                }
            }

            loop {
                let mut done = 0;
                let mut size = 0;
                let ret = test(send_req, &mut done, &mut size);
                assert_eq!(ret, ncclResult_t::ncclSuccess);
                if done == 1 {
                    assert_eq!(size, data.len().try_into().unwrap());
                    break;
                }
            }

            assert_eq!(buf[..data.len()], data);
        }
    }
}
