use core::slice;
use nccl_net_sys::ncclDebugLogSubSys as sys;
use nccl_net_sys::*;
use roma::{consts::HomaRecvmsgFlags, HomaSocket};
use socket2::{Domain, SockAddr};
use std::{
    ffi::CString,
    ffi::{c_int, c_void},
    io::ErrorKind,
    net::{IpAddr, Ipv4Addr, SocketAddr, ToSocketAddrs},
    ptr::null_mut,
};

enum Request<'a, 'b> {
    Send(SendRequest<'a>),
    Recv(RecvRequest<'a, 'b>),
}

struct SendRequest<'a> {
    comm: &'a mut SendComm,
    id: u64,
}

struct RecvRequest<'a, 'b> {
    comm: &'a mut RecvComm,
    buffer: &'b mut [u8],
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

struct Homa {}

impl Homa {
    pub fn devices(ndev: &mut c_int) -> ncclResult_t {
        *ndev = 1;

        ncclResult_t::ncclSuccess
    }
    pub fn get_properties(dev: c_int, props: &mut ncclNetProperties_v6_t) -> ncclResult_t {
        if dev == 0 {
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
        } else {
            ncclResult_t::ncclInternalError
        }
    }

    pub fn listen(dev: c_int, handle: &mut SockAddr) -> (ListenComm, ncclResult_t) {
        assert_eq!(dev, 0);

        let addr = if_addrs::get_if_addrs()
            .unwrap()
            .iter()
            .map(|i| i.addr.ip())
            .find(|a| {
                if let IpAddr::V4(v4) = a {
                    v4.is_private()
                } else {
                    false
                }
            })
            .unwrap_or_else(|| IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));

        let socket = HomaSocket::new(Domain::IPV4, 1000).unwrap();

        socket
            .socket
            .bind(&(addr, 0).to_socket_addrs().unwrap().next().unwrap().into())
            .unwrap();

        let local_port = socket
            .socket
            .local_addr()
            .unwrap()
            .as_socket()
            .unwrap()
            .port();

        *handle = SocketAddr::new(addr, local_port).into();

        let comm = ListenComm {
            socket: Some(socket),
        };

        (comm, ncclResult_t::ncclSuccess)
    }

    pub fn connect(dev: c_int, handle: &SocketAddr) -> (SendComm, ncclResult_t) {
        assert_eq!(dev, 0);

        let socket = HomaSocket::new(Domain::IPV4, 1000).unwrap();
        let comm = SendComm {
            socket,
            remote: handle.clone(),
        };

        (comm, ncclResult_t::ncclSuccess)
    }

    pub fn accept(listen_comm: &mut ListenComm) -> (RecvComm, ncclResult_t) {
        let comm = RecvComm {
            socket: listen_comm.socket.take().unwrap(),
        };
        (comm, ncclResult_t::ncclSuccess)
    }

    pub fn reg_mr(
        _comm: *mut c_void,
        _data: *mut c_void,
        _size: c_int,
        type_: c_int,
        _mhandle: *mut *mut c_void,
    ) -> ncclResult_t {
        if type_ != NCCL_PTR_HOST as i32 {
            ncclResult_t::ncclInternalError
        } else {
            ncclResult_t::ncclSuccess
        }
    }

    pub fn dereg_mr(_comm: *mut c_void, _mhandle: *mut c_void) -> ncclResult_t {
        ncclResult_t::ncclSuccess
    }

    pub fn isend<'a, 'b, 'c>(
        send_comm: &'a mut SendComm,
        buf: &'b [u8],
    ) -> (Request<'a, 'c>, ncclResult_t) {
        let id = send_comm
            .socket
            .send(
                buf,
                send_comm.remote.into(),
                0,
                buf.len().try_into().unwrap(),
            )
            .unwrap();
        (
            Request::Send(SendRequest {
                comm: send_comm,
                id,
            }),
            ncclResult_t::ncclSuccess,
        )
    }

    pub fn irecv<'a, 'b>(
        recv_comm: &'a mut RecvComm,
        buf: &'b mut [u8],
    ) -> (Request<'a, 'b>, ncclResult_t) {
        (
            Request::Recv(RecvRequest {
                buffer: buf,
                comm: recv_comm,
            }),
            ncclResult_t::ncclSuccess,
        )
    }

    pub extern "C" fn test(
        request: &mut Request,
        done: &mut c_int,
        sizes: Option<&mut c_int>,
    ) -> ncclResult_t {
        match request {
            Request::Send(req) => {
                match req
                    .comm
                    .socket
                    .recv(&mut [], HomaRecvmsgFlags::NONBLOCKING, req.id)
                {
                    Ok((_, _, _, cookie)) => {
                        *done = 1;
                        if let Some(size) = sizes {
                            *size = cookie.try_into().unwrap();
                        }
                        // FIXME: drop request handle
                        ncclResult_t::ncclSuccess
                    }
                    Err(err) if err.kind() == ErrorKind::WouldBlock => {
                        *done = 0;
                        ncclResult_t::ncclSuccess
                    }
                    Err(err) => panic!("{}", err),
                }
            }
            Request::Recv(req) => match req.comm.socket.recv(
                req.buffer,
                HomaRecvmsgFlags::REQUEST | HomaRecvmsgFlags::NONBLOCKING,
                0,
            ) {
                Ok((length, addr, id, _)) => {
                    *done = 1;
                    if let Some(size) = sizes {
                        *size = length.try_into().unwrap();
                    }
                    // FIXME: drop request handle
                    req.comm.socket.send(&[], addr, id, 0).unwrap();
                    ncclResult_t::ncclSuccess
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => {
                    *done = 0;
                    ncclResult_t::ncclSuccess
                }
                Err(err) => panic!("{}", err),
            },
        }
    }

    pub fn close_send(send_comm: SendComm) -> ncclResult_t {
        ncclResult_t::ncclSuccess
    }

    pub fn close_recv(recv_comm: RecvComm) -> ncclResult_t {
        ncclResult_t::ncclSuccess
    }

    pub fn close_listen(listen_comm: ListenComm) -> ncclResult_t {
        ncclResult_t::ncclSuccess
    }
}
