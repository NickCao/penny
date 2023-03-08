use roma::{consts::HomaRecvmsgFlags, *};
use socket2::Domain;
use std::net::SocketAddr;

fn main() {
    env_logger::init();

    let mut socket = HomaSocket::new(Domain::IPV4, 1000).unwrap();

    let listen: SocketAddr = "127.0.0.1:4000".parse().unwrap();
    socket.socket.bind(&listen.into()).unwrap();

    let mut bufs = vec![0u8; consts::HOMA_MAX_MESSAGE_LENGTH];

    loop {
        match socket.recv(&mut bufs, HomaRecvmsgFlags::REQUEST, 0) {
            Ok((length, addr, id, _)) => {
                socket.send(&bufs[..length], addr, id, 0).unwrap();
            }
            Err(err) => panic!("{}", err),
        }
    }
}
