#![feature(int_roundings)]
#![feature(default_free_fn)]

use memmap2::{MmapMut, MmapOptions};
use nix::sys::socket::setsockopt;
use socket2::{Domain, SockAddr, Socket, Type};
use std::cmp::min;
use std::collections::VecDeque;
use std::ffi::c_int;
use std::io::{Error, ErrorKind, IoSlice, Result};
use std::mem::size_of_val;
use std::net::SocketAddr;
use std::os::fd::AsRawFd;
use std::slice;

pub mod consts;
pub mod types;

pub struct HomaSocket {
    pub socket: Socket,
    buffer: MmapMut,
    backlog: VecDeque<u32>,
}

impl HomaSocket {
    pub fn new(domain: Domain, pages: usize) -> Result<Self> {
        log::debug!("HomaSocket::new(domain: {:?}, pages: {})", domain, pages);

        let socket = Socket::new_raw(domain, Type::DGRAM, Some(consts::IPPROTO_HOMA.into()))?;

        let length = pages * consts::HOMA_BPAGE_SIZE;
        let buffer = MmapOptions::new().len(length).map_anon()?;

        setsockopt(socket.as_raw_fd(), types::HomaBuf, &buffer).unwrap();

        Ok(Self {
            socket,
            buffer,
            backlog: VecDeque::default(),
        })
    }

    pub fn send(
        &self,
        buf: &[u8],
        addr: SocketAddr,
        id: u64,
        completion_cookie: u64,
    ) -> Result<u64> {
        log::debug!(
            "HomaSocket::send(buf.len(): {}, addr: {}, id: {}, completion_cookie: {})",
            buf.len(),
            addr,
            id,
            completion_cookie
        );

        let addr = SockAddr::from(addr);

        let pad = vec![0];
        let iov = vec![IoSlice::new(buf), IoSlice::new(&pad)];

        let mut sendmsg_args = types::homa_sendmsg_args {
            id,
            completion_cookie,
        };

        let mut hdr = libc::msghdr {
            msg_name: addr.as_ptr() as *mut _,
            msg_namelen: addr.len(),
            msg_iov: iov.as_ptr() as *mut _,
            msg_iovlen: iov.len(),
            msg_control: (&mut sendmsg_args as *mut types::homa_sendmsg_args).cast(),
            msg_controllen: 0,
            msg_flags: 0,
        };

        let result = unsafe { libc::sendmsg(self.socket.as_raw_fd(), &mut hdr, 0) };

        if result < 0 {
            return Err(Error::last_os_error());
        }

        Ok(sendmsg_args.id)
    }

    pub fn recv(
        &mut self,
        buf: &mut [u8],
        flags: consts::HomaRecvmsgFlags,
        id: u64,
    ) -> Result<(usize, SocketAddr, u64, u64)> {
        log::debug!(
            "HomaSocket::recv(buf.len(): {}, flags: {:?}, id: {})",
            buf.len(),
            flags,
            id,
        );

        let num_bpages = min(self.backlog.len(), consts::HOMA_MAX_BPAGES);
        let bpages: Vec<u32> = self.backlog.drain(0..num_bpages).collect();

        let mut bpage_offsets = [0; consts::HOMA_MAX_BPAGES];
        bpage_offsets[..bpages.len()].copy_from_slice(&bpages);

        let mut recvmsg_args = types::homa_recvmsg_args {
            id,
            completion_cookie: 0,
            flags: flags.bits(),
            num_bpages: num_bpages.try_into().unwrap(),
            pad: [0; 2],
            bpage_offsets,
        };

        let mut addr: libc::sockaddr_storage = unsafe { std::mem::zeroed() };

        let mut hdr = libc::msghdr {
            msg_name: (&mut addr as *mut libc::sockaddr_storage).cast(),
            msg_namelen: size_of_val(&addr).try_into().unwrap(),
            msg_iov: std::ptr::null_mut(),
            msg_iovlen: 0,
            msg_control: (&mut recvmsg_args as *mut types::homa_recvmsg_args).cast(),
            msg_controllen: size_of_val(&recvmsg_args),
            msg_flags: 0,
        };

        let length = unsafe {
            libc::recvmsg(
                self.socket.as_raw_fd(),
                &mut hdr,
                0, // flags are ignored
            )
        };

        if length < 0 {
            return Err(Error::last_os_error());
        }

        let length: usize = length.try_into().unwrap();

        if buf.len() < length - 1 {
            return Err(Error::new(ErrorKind::OutOfMemory, "buffer too small"));
        }

        for i in 0..recvmsg_args.num_bpages as usize {
            let (len, last) = if i != recvmsg_args.num_bpages as usize - 1 {
                (consts::HOMA_BPAGE_SIZE, 0)
            } else {
                (
                    length - consts::HOMA_BPAGE_SIZE * (recvmsg_args.num_bpages as usize - 1),
                    1,
                )
            };
            let offset = recvmsg_args.bpage_offsets[i];
            unsafe {
                self.backlog.push_back(offset);
                let data = self.buffer.as_ptr().offset(offset.try_into().unwrap());
                buf[i * consts::HOMA_BPAGE_SIZE..i * consts::HOMA_BPAGE_SIZE + len - last]
                    .copy_from_slice(slice::from_raw_parts(data, len - last));
            }
        }

        let addr = unsafe { SockAddr::new(addr, size_of_val(&addr).try_into().unwrap()) };

        Ok((
            length - 1,
            addr.as_socket().unwrap(),
            recvmsg_args.id,
            recvmsg_args.completion_cookie,
        ))
    }

    pub fn abort(&self, id: u64, error: c_int) -> nix::Result<i32> {
        let mut abort_args = types::homa_abort_args::new(id, error);
        unsafe { types::homa_abort(self.socket.as_raw_fd(), &mut abort_args) }
    }

    pub fn freeze(&self) -> nix::Result<i32> {
        unsafe { types::homa_freeze(self.socket.as_raw_fd()) }
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use rand::{RngCore, SeedableRng};

    #[test]
    fn roundtrip() {
        let _server = std::thread::spawn(|| {
            let mut socket = HomaSocket::new(Domain::IPV4, 1000).unwrap();

            let addr: SocketAddr = "127.0.0.1:4000".parse().unwrap();

            socket.socket.bind(&addr.into()).unwrap();

            let mut bufs = vec![0u8; consts::HOMA_MAX_MESSAGE_LENGTH];

            loop {
                match socket.recv(&mut bufs, consts::HomaRecvmsgFlags::REQUEST, 0) {
                    Ok((length, addr, id, _)) => {
                        socket.send(&bufs[..length], addr, id, 0).unwrap();
                    }
                    Err(err) => panic!("{}", err),
                }
            }
        });

        let client = std::thread::spawn(|| {
            let mut socket = HomaSocket::new(Domain::IPV4, 1000).unwrap();

            let addr: SocketAddr = "127.0.0.1:4000".parse().unwrap();

            let mut buf = vec![0u8; consts::HOMA_MAX_MESSAGE_LENGTH];

            let mut i = 1;

            while i < consts::HOMA_MAX_MESSAGE_LENGTH {
                let mut rng = rand::rngs::StdRng::seed_from_u64(i.try_into().unwrap());

                let mut src = vec![0u8; i];

                rng.fill_bytes(&mut src);

                let id = socket.send(&src, addr.into(), 0, 0).unwrap();

                let (length, _, _, _) = socket
                    .recv(&mut buf, consts::HomaRecvmsgFlags::empty(), id)
                    .unwrap();

                assert_eq!(src.len(), length);
                assert_eq!(src, buf[..length]);

                i *= 2
            }
        });

        client.join().unwrap();
    }
}
