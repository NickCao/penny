use std::{default::default, mem::size_of};

use crate::consts;
use libc::{c_int, c_void, size_t, socklen_t};
use memmap2::MmapMut;
use nix::{errno::Errno, sys::socket::SetSockOpt};

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Clone)]
struct homa_set_buf_args {
    pub start: *mut c_void,
    pub length: size_t,
}

impl From<&MmapMut> for homa_set_buf_args {
    fn from(value: &MmapMut) -> Self {
        Self {
            start: value.as_ptr().cast_mut().cast(),
            length: value.len(),
        }
    }
}

#[derive(Clone)]
pub struct HomaBuf;

impl SetSockOpt for HomaBuf {
    type Val = MmapMut;

    fn set(&self, fd: std::os::fd::RawFd, val: &Self::Val) -> nix::Result<()> {
        unsafe {
            let args: homa_set_buf_args = val.into();

            let res = libc::setsockopt(
                fd,
                consts::IPPROTO_HOMA,
                consts::SO_HOMA_SET_BUF,
                &args as *const homa_set_buf_args as *const c_void,
                size_of::<homa_set_buf_args>() as socklen_t,
            );

            Errno::result(res).map(drop)
        }
    }
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug)]
pub struct homa_sendmsg_args {
    pub id: u64,
    pub completion_cookie: u64,
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug)]
pub struct homa_recvmsg_args {
    pub id: u64,
    pub completion_cookie: u64,
    pub flags: c_int,
    pub num_bpages: u32,
    pub pad: [u32; 2],
    pub bpage_offsets: [u32; consts::HOMA_MAX_BPAGES],
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug)]
pub struct homa_abort_args {
    pub id: u64,
    pub error: c_int,
    pub pad1: c_int,
    pub pad2: [u64; 2],
}

impl homa_abort_args {
    pub fn new(id: u64, error: c_int) -> Self {
        Self {
            id,
            error,
            pad1: default(),
            pad2: default(),
        }
    }
}

nix::ioctl_readwrite!(homa_abort, 0x89, 0xe3, homa_abort_args);
nix::ioctl_none!(homa_freeze, 0x89, 0xef);
