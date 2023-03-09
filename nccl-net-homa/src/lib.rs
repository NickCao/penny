#![feature(ip)]
#![feature(cstr_from_bytes_until_nul)]
#![feature(strict_provenance)]

use nccl_net_sys::ncclNet_v6_t;

mod binding;
pub mod error;
pub mod homa;
pub mod logger;

#[export_name = "ncclNetPlugin_v6"]
static mut PLUGIN: ncclNet_v6_t = ncclNet_v6_t {
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
