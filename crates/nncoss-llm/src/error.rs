use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    InvalidArgument(&'static str),
    InvalidModel(&'static str),
    Unsupported(&'static str),
    ContextLength { requested: usize, maximum: usize },
    TokenOutOfRange(u32),
    NoLogits,
    OutputTooSmall { expected: usize, actual: usize },
    NonFiniteLogit,
    Allocation,
    Panic,
}

impl Error {
    pub(crate) fn code(&self) -> i32 {
        match self {
            Self::InvalidArgument(_) => -1,
            Self::InvalidModel(_) => -2,
            Self::Unsupported(_) => -3,
            Self::ContextLength { .. } => -4,
            Self::TokenOutOfRange(_) => -5,
            Self::NoLogits => -6,
            Self::OutputTooSmall { .. } => -7,
            Self::NonFiniteLogit => -8,
            Self::Allocation => -9,
            Self::Panic => -99,
        }
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(formatter, "参数错误：{message}"),
            Self::InvalidModel(message) => write!(formatter, "模型格式错误：{message}"),
            Self::Unsupported(message) => write!(formatter, "不支持：{message}"),
            Self::ContextLength { requested, maximum } => {
                write!(formatter, "上下文长度 {requested} 超过上限 {maximum}")
            }
            Self::TokenOutOfRange(token) => write!(formatter, "token {token} 超出词表范围"),
            Self::NoLogits => write!(formatter, "尚未执行 prefill 或 decode"),
            Self::OutputTooSmall { expected, actual } => {
                write!(
                    formatter,
                    "输出缓冲区需要 {expected} 个元素，实际为 {actual}"
                )
            }
            Self::NonFiniteLogit => write!(formatter, "logits 包含非有限值"),
            Self::Allocation => write!(formatter, "内存分配失败或请求尺寸过大"),
            Self::Panic => write!(formatter, "运行时发生内部 panic"),
        }
    }
}

impl std::error::Error for Error {}
