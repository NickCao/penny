use rand::{RngCore, SeedableRng};
use roma::{consts::HomaRecvmsgFlags, *};
use socket2::Domain;
use std::net::SocketAddr;

fn main() {
    env_logger::init();

    let mut socket = HomaSocket::new(Domain::IPV4, 1000).unwrap();
    let dest: SocketAddr = "127.0.0.1:4000".parse().unwrap();

    let mut buf = vec![0u8; consts::HOMA_MAX_MESSAGE_LENGTH];

    let mut i = 1;
    while i < consts::HOMA_MAX_MESSAGE_LENGTH {
        let mut rng = rand::rngs::StdRng::seed_from_u64(i.try_into().unwrap());
        let mut src = vec![0u8; i];
        rng.fill_bytes(&mut src);

        let id = socket.send(&src, dest.into(), 0, 0).unwrap();

        let (length, _, _, _) = socket
            .recv(&mut buf, HomaRecvmsgFlags::empty(), id)
            .unwrap();

        assert_eq!(src.len(), length);
        assert_eq!(src, buf[..length]);

        i *= 2
    }
}
