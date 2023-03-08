#![feature(c_variadic)]

use nccl_net_homa::homa::*;
use nccl_net_sys::*;
use socket2::SockAddr;
use std::{ffi::CStr, os::raw::*};

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

fn main() {
    let ret = nccl_net_homa::init(Some(logger));
    assert_eq!(ret, ncclResult_t::ncclSuccess);

    let mut handle = unsafe { SockAddr::try_init(|_, _| Ok(())).unwrap().1 };
    let (mut listen_comm, result) = Homa::listen(0, &mut handle);
    assert_eq!(result, ncclResult_t::ncclSuccess);

    let (mut send_comm, result) = Homa::connect(0, &handle);
    assert_eq!(result, ncclResult_t::ncclSuccess);

    let (mut recv_comm, result) = Homa::accept(&mut listen_comm);
    assert_eq!(result, ncclResult_t::ncclSuccess);

    let data = b"hello penny\0".to_vec();
    let (mut send_req, result) = Homa::isend(&mut send_comm, &data);
    assert_eq!(result, ncclResult_t::ncclSuccess);

    let mut buf = vec![0u8; 100];
    let (mut recv_req, result) = Homa::irecv(&mut recv_comm, &mut buf);
    assert_eq!(result, ncclResult_t::ncclSuccess);

    loop {
        let mut done = 0;
        let mut size = 0;
        let ret = Homa::test(&mut recv_req, &mut done, Some(&mut size));
        assert_eq!(ret, ncclResult_t::ncclSuccess);
        if done == 1 {
            assert_eq!(size, data.len().try_into().unwrap());
            break;
        }
    }

    loop {
        let mut done = 0;
        let mut size = 0;
        let ret = Homa::test(&mut send_req, &mut done, Some(&mut size));
        assert_eq!(ret, ncclResult_t::ncclSuccess);
        if done == 1 {
            assert_eq!(size, data.len().try_into().unwrap());
            break;
        }
    }

    assert_eq!(buf[..data.len()], data);
}
