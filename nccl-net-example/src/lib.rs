use core::slice;
use nccl_net_sys::ncclDebugLogSubSys as sys;
use nccl_net_sys::*;
use roma::{consts::HomaRecvmsgFlags, HomaSocket};
use socket2::Domain;
use std::{
    ffi::CString,
    ffi::{c_int, c_void},
    io::{ErrorKind, IoSlice},
    net::{SocketAddr, ToSocketAddrs},
    ptr::null_mut,
};

const PLACEHOLDER: &[u8] = &[1];

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

enum Request<'a, 'b, 'c> {
    Send(SendRequest<'a, 'b>),
    Recv(RecvRequest<'a, 'b, 'c>),
}

struct SendRequest<'a, 'b> {
    comm: &'a mut SendComm<'b>,
    id: u64,
    size: usize,
}

struct RecvRequest<'a, 'b, 'c> {
    comm: &'a mut RecvComm<'b>,
    buffer: &'c mut [u8],
}

struct ListenComm {
    socket: Option<HomaSocket>,
}

struct SendComm<'a> {
    socket: HomaSocket,
    buffer: Vec<IoSlice<'a>>,
    remote: SocketAddr,
}

struct RecvComm<'a> {
    socket: HomaSocket,
    buffer: Vec<IoSlice<'a>>,
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
    let comm = Box::new(SendComm {
        socket,
        remote,
        buffer: vec![],
    });
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
        buffer: vec![],
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
    tag: c_int,
    _mhandle: *mut c_void,
    request: *mut *mut c_void,
) -> ncclResult_t {
    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET,
        "homa::isend(data: {:?}, size: {}, tag: {})",
        data,
        size,
        tag
    );

    let comm = &mut *(send_comm as *mut SendComm);

    let size: usize = size.try_into().unwrap();
    let data = IoSlice::new(slice::from_raw_parts(data.cast(), size));

    let id = comm.socket.send(comm.remote, &[data], 0, 0).unwrap();

    *request = Box::into_raw(Box::new(Request::Send(SendRequest { id, comm, size }))).cast();

    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn irecv(
    recv_comm: *mut c_void,
    n: c_int,
    data: *mut *mut c_void,
    sizes: *mut c_int,
    tags: *mut c_int,
    _mhandles: *mut *mut c_void,
    request: *mut *mut c_void,
) -> ncclResult_t {
    let n: usize = n.try_into().unwrap();
    let data = slice::from_raw_parts(data, n);
    let sizes = slice::from_raw_parts(sizes, n);
    let tags = slice::from_raw_parts(tags, n);

    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET,
        "homa::irecv(data: {:?}, sizes: {:?}, tags: {:?})",
        data,
        sizes,
        tags,
    );

    if n != 1 {
        return ncclResult_t::ncclInternalError;
    }

    let comm = &mut *(recv_comm as *mut RecvComm);

    let buffer = slice::from_raw_parts_mut(data[0].cast(), sizes[0].try_into().unwrap());

    *request = Box::into_raw(Box::new(Request::Recv(RecvRequest { buffer, comm }))).cast();

    ncclResult_t::ncclSuccess
}

unsafe extern "C" fn test(
    request: *mut c_void,
    done: *mut c_int,
    sizes: *mut c_int,
) -> ncclResult_t {
    let request: &mut Request = &mut *request.cast();

    log!(
        ncclDebugLogLevel::NCCL_LOG_TRACE,
        sys::NCCL_NET,
        "homa::irecv(request: TODO)",
    );

    match request {
        Request::Send(req) => match req.comm.socket.recv(
            req.id,
            HomaRecvmsgFlags::RESPONSE | HomaRecvmsgFlags::NONBLOCKING,
            &req.comm.buffer,
        ) {
            Ok((_, _, buffer, _)) => {
                *done = 1;
                *sizes = req.size.try_into().unwrap();
                req.comm.buffer = buffer;
                // FIXME: drop request handle
                return ncclResult_t::ncclSuccess;
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                *done = 0;
                return ncclResult_t::ncclSuccess;
            }
            Err(err) => panic!("{}", err),
        },
        Request::Recv(req) => match req.comm.socket.recv(
            0,
            HomaRecvmsgFlags::REQUEST | HomaRecvmsgFlags::NONBLOCKING,
            &req.comm.buffer,
        ) {
            Ok((id, _, buffer, addr)) => {
                *done = 1;
                *sizes = buffer
                    .iter()
                    .map(|buf| buf.len())
                    .sum::<usize>()
                    .try_into()
                    .unwrap();
                let mut cur: usize = 0;
                for buf in &buffer {
                    req.buffer[cur..cur + buf.len()].copy_from_slice(buf);
                    cur += buf.len();
                }
                req.comm.buffer = buffer;
                // FIXME: drop request handle
                req.comm
                    .socket
                    .send(addr.unwrap(), &[IoSlice::new(PLACEHOLDER)], id, 0)
                    .unwrap();
                return ncclResult_t::ncclSuccess;
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                *done = 0;
                return ncclResult_t::ncclSuccess;
            }
            Err(err) => panic!("{}", err),
        },
    }
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
