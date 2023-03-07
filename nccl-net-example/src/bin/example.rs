#![feature(c_variadic)]

use nccl_net_example::*;
use nccl_net_sys::*;
use std::{
    ffi::{c_void, CStr},
    os::raw::*,
    ptr::{null, null_mut},
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

fn main() {
    unsafe {
        let ret = init(Some(logger));
        assert_eq!(ret, ncclResult_t::ncclSuccess);
        let mut ndev = 0;
        let ret = devices(&mut ndev);
        assert_eq!(ret, ncclResult_t::ncclSuccess);
        for dev in 0..ndev {
            let mut props = std::mem::zeroed();
            let ret = get_properties(dev, &mut props);
            assert_eq!(ret, ncclResult_t::ncclSuccess);
        }
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
        let mut buf = vec![0u8, 100];
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
    }
}
