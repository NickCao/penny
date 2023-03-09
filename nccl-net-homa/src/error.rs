use nccl_net_sys::ncclResult_t;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("system")]
    System(#[from] std::io::Error),
    #[error("internal")]
    Internal,
    #[error("invalid usage")]
    InvalidUsage,
    #[error("invalid argument")]
    InvalidArgument,
}

impl From<Error> for ncclResult_t {
    fn from(val: Error) -> Self {
        match val {
            Error::System(_) => ncclResult_t::ncclSystemError,
            Error::Internal => ncclResult_t::ncclInternalError,
            Error::InvalidUsage => ncclResult_t::ncclInvalidUsage,
            Error::InvalidArgument => ncclResult_t::ncclInvalidArgument,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
