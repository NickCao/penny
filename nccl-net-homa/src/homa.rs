use crate::error::{Error, Result};
use log::LevelFilter;
use nccl_net_sys::*;
use roma::{consts::HomaRecvmsgFlags, HomaSocket};
use socket2::Domain;
use std::{
    ffi::{c_int, CStr, CString},
    io::ErrorKind,
    net::{IpAddr, Ipv4Addr, SocketAddr, ToSocketAddrs},
    ptr::null_mut,
};

pub enum Request<'a, 'b> {
    Send(SendRequest<'a>),
    Recv(RecvRequest<'a, 'b>),
}

pub struct SendRequest<'a> {
    comm: &'a mut SendComm,
    id: u64,
}

pub struct RecvRequest<'a, 'b> {
    comm: &'a mut RecvComm,
    buffer: &'b mut [u8],
}

pub struct ListenComm {
    socket: Option<HomaSocket>,
}

pub struct SendComm {
    socket: HomaSocket,
    remote: SocketAddr,
    inflight: bool,
}

pub struct RecvComm {
    socket: HomaSocket,
}

pub struct Homa {}

impl Homa {
    pub fn init(logger: ncclDebugLogger_t) -> Result<()> {
        crate::logger::Logger::init(LevelFilter::Debug, logger).unwrap();
        Ok(())
    }

    pub fn devices() -> Result<i32> {
        Ok(1)
    }

    pub fn get_properties(dev: i32) -> Result<ncclNetProperties_v6_t> {
        if dev == 0 {
            let name = Box::leak(Box::new(CString::new("default").unwrap()));
            Ok(ncclNetProperties_v6_t {
                name: name.as_ptr().cast_mut(),
                pciPath: null_mut(),
                guid: 0,
                ptrSupport: NCCL_PTR_HOST as i32,
                speed: 1000,
                port: 0,
                latency: 0.0,
                maxComms: i32::MAX,
                maxRecvs: 1,
            })
        } else {
            Err(Error::Internal)
        }
    }

    pub fn listen(dev: i32, handle: &mut [u8]) -> Result<ListenComm> {
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

        let socket = HomaSocket::new(Domain::IPV4, 20000).unwrap();

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

        let h = CString::new(SocketAddr::new(addr, local_port).to_string()).unwrap();
        let hr = h.as_bytes_with_nul();
        handle[..hr.len()].copy_from_slice(hr);

        Ok(ListenComm {
            socket: Some(socket),
        })
    }

    pub fn connect(dev: c_int, handle: &[u8]) -> Result<SendComm> {
        assert_eq!(dev, 0);

        let handle = CStr::from_bytes_until_nul(handle)
            .unwrap()
            .to_str()
            .unwrap();

        let socket = HomaSocket::new(Domain::IPV4, 20000).unwrap();

        Ok(SendComm {
            socket,
            remote: handle.parse().unwrap(),
            inflight: false,
        })
    }

    pub fn accept(listen_comm: &mut ListenComm) -> Result<RecvComm> {
        Ok(RecvComm {
            socket: listen_comm.socket.take().unwrap(),
        })
    }

    pub fn isend<'a, 'b>(
        send_comm: &'a mut SendComm,
        buf: &[u8],
    ) -> Result<Option<Request<'a, 'b>>> {
        if send_comm.inflight {
            return Ok(None);
        }

        send_comm.inflight = true;

        let id = send_comm
            .socket
            .send(buf, send_comm.remote, 0, buf.len().try_into().unwrap())?;

        Ok(Some(Request::Send(SendRequest {
            comm: send_comm,
            id,
        })))
    }

    pub fn irecv<'a, 'b>(
        recv_comm: &'a mut RecvComm,
        buf: &'b mut [u8],
    ) -> Result<Request<'a, 'b>> {
        Ok(Request::Recv(RecvRequest {
            buffer: buf,
            comm: recv_comm,
        }))
    }

    pub fn test(request: &mut Request) -> Result<Option<i32>> {
        match request {
            Request::Send(req) => {
                match req
                    .comm
                    .socket
                    .recv(&mut [], HomaRecvmsgFlags::NONBLOCKING, req.id)
                {
                    Ok((_, _, _, cookie)) => {
                        req.comm.inflight = false;
                        Ok(Some(cookie.try_into().unwrap()))
                    }
                    Err(err) if err.kind() == ErrorKind::WouldBlock => Ok(None),
                    Err(err) => Err(err)?,
                }
            }
            Request::Recv(req) => match req.comm.socket.recv(
                req.buffer,
                HomaRecvmsgFlags::REQUEST | HomaRecvmsgFlags::NONBLOCKING,
                0,
            ) {
                Ok((length, addr, id, _)) => {
                    req.comm.socket.send(&[], addr, id, 0).unwrap();
                    Ok(Some(length.try_into().unwrap()))
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => Ok(None),
                Err(err) => Err(err)?,
            },
        }
    }

    pub fn close_send(_send_comm: SendComm) -> Result<()> {
        Ok(())
    }

    pub fn close_recv(_recv_comm: RecvComm) -> Result<()> {
        Ok(())
    }

    pub fn close_listen(_listen_comm: ListenComm) -> Result<()> {
        Ok(())
    }
}
