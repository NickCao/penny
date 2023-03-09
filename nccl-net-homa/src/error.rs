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

impl Into<ncclResult_t> for Error {
    fn into(self) -> ncclResult_t {
        match self {
            Self::System(_) => ncclResult_t::ncclSystemError,
            Self::Internal => ncclResult_t::ncclInternalError,
            Self::InvalidUsage => ncclResult_t::ncclInvalidUsage,
            Self::InvalidArgument => ncclResult_t::ncclInvalidArgument,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
