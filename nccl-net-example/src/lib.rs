use core::slice;
use nccl_net_sys::ncclDebugLogSubSys as sys;
use nccl_net_sys::*;
use roma::HomaSocket;
use socket2::Domain;
use std::{
    ffi::CString,
    ffi::{c_int, c_void},
    io::IoSlice,
    net::{SocketAddr, ToSocketAddrs},
    ptr::null_mut,
};

static mut LOGGER: ncclDebugLogger_t = None;

macro_rules! log {
    ($level:expr, $sys:expr, $($arg:tt)*) => {
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
    };
}

struct Request {
    id: u64,
}

struct ListenComm {
    socket: Option<HomaSocket>,
}

struct SendComm {
    socket: HomaSocket,
    remote: SocketAddr,
}

struct RecvComm {
    socket: HomaSocket,
}

unsafe extern "C" fn init(logger: ncclDebugLogger_t) -> ncclResult_t {
    LOGGER = logger;
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET | sys::NCCL_INIT,
        "init",
    );
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn devices(ndev: *mut c_int) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET | sys::NCCL_INIT,
        "devices",
    );
    *ndev = 1;
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn get_properties(
    dev: c_int,
    props: *mut ncclNetProperties_v6_t,
) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET | sys::NCCL_INIT,
        "get_properties",
    );
    assert_eq!(dev, 0);
    *props = ncclNetProperties_v6_t {
        name: "default\0".as_ptr().cast_mut().cast(),
        pciPath: null_mut(),
        guid: 0,
        ptrSupport: NCCL_PTR_HOST as i32,
        speed: 1000,
        port: 0,
        latency: 0.0,
        maxComms: i32::MAX,
        maxRecvs: 1,
    };
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn listen(
    dev: c_int,
    handle: *mut c_void,
    listen_comm: *mut *mut c_void,
) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET | sys::NCCL_INIT,
        "listen",
    );
    assert_eq!(dev, 0);
    let socket = HomaSocket::new(Domain::IPV4, 1000).unwrap();
    socket
        .socket
        .bind(
            &("127.0.0.1", 0)
                .to_socket_addrs()
                .unwrap()
                .next()
                .unwrap()
                .into(),
        )
        .unwrap();
    let local = socket.socket.local_addr().unwrap().as_socket().unwrap();
    *(handle as *mut SocketAddr) = local;
    let comm = Box::new(ListenComm {
        socket: Some(socket),
    });
    *listen_comm = Box::into_raw(comm).cast();
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn connect(
    dev: c_int,
    handle: *mut c_void,
    send_comm: *mut *mut c_void,
) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET | sys::NCCL_INIT,
        "connect",
    );
    assert_eq!(dev, 0);
    let socket = HomaSocket::new(Domain::IPV4, 1000).unwrap();
    let remote = *(handle as *const SocketAddr);
    let comm = Box::new(SendComm { socket, remote });
    *send_comm = Box::into_raw(comm).cast();
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn accept(listen_comm: *mut c_void, recv_comm: *mut *mut c_void) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET | sys::NCCL_INIT,
        "accept",
    );
    let listen_comm = &mut *(listen_comm as *mut ListenComm);
    let comm = Box::new(RecvComm {
        socket: listen_comm.socket.take().unwrap(),
    });
    *recv_comm = Box::into_raw(comm).cast();
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn reg_mr(
    _comm: *mut c_void,
    _data: *mut c_void,
    _size: c_int,
    type_: c_int,
    _mhandle: *mut *mut c_void,
) -> ncclResult_t {
    log!(ncclDebugLogLevel::NCCL_LOG_TRACE, sys::NCCL_NET, "reg_mr");
    if type_ != NCCL_PTR_HOST as i32 {
        ncclResult_t::ncclInternalError
    } else {
        ncclResult_t::ncclSuccess
    }
}

unsafe extern "C" fn dereg_mr(_comm: *mut c_void, _mhandle: *mut c_void) -> ncclResult_t {
    log!(ncclDebugLogLevel::NCCL_LOG_TRACE, sys::NCCL_NET, "dereg_mr");
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
    log!(ncclDebugLogLevel::NCCL_LOG_TRACE, sys::NCCL_NET, "isend");
    let comm = &*(send_comm as *mut SendComm);
    let id = comm
        .socket
        .send(
            comm.remote,
            &[IoSlice::new(slice::from_raw_parts(
                data.cast(),
                size.try_into().unwrap(),
            ))],
            0,
            0,
        )
        .unwrap();
    *request = Box::into_raw(Box::new(Request { id })).cast();
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn irecv(
    recv_comm: *mut c_void,
    _n: c_int,
    _data: *mut *mut c_void,
    _sizes: *mut c_int,
    _tags: *mut c_int,
    _mhandles: *mut *mut c_void,
    _request: *mut *mut c_void,
) -> ncclResult_t {
    log!(ncclDebugLogLevel::NCCL_LOG_TRACE, sys::NCCL_NET, "irecv");
    let _comm = &*(recv_comm as *mut RecvComm);
    // comm.socket.recv(id, flags, bufs);
    ncclResult_t::ncclInternalError
}

unsafe extern "C" fn test(
    _request: *mut c_void,
    _done: *mut c_int,
    _sizes: *mut c_int,
) -> ncclResult_t {
    log!(ncclDebugLogLevel::NCCL_LOG_TRACE, sys::NCCL_NET, "test");
    ncclResult_t::ncclInternalError
}

unsafe extern "C" fn close_send(send_comm: *mut c_void) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET,
        "close_send"
    );
    drop(Box::from_raw(send_comm as *mut SendComm));
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn close_recv(recv_comm: *mut c_void) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET,
        "close_recv"
    );
    drop(Box::from_raw(recv_comm as *mut RecvComm));
    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn close_listen(listen_comm: *mut c_void) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET,
        "close_listen"
    );
    drop(Box::from_raw(listen_comm as *mut ListenComm));
    ncclResult_t::ncclSuccess
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
