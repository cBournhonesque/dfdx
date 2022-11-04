use std::vec::Vec;
use crate::tensor::*;


// Collate takes a group of individual samples and builds a batch from them
pub trait Collate<T>: Default + Clone {
    // type of collate function's output = type of a batch
    // The batch size B is fixed
    type Output<const B: usize>;

    fn collate(&self, samples: [T; B]) -> Self::Output<{ B }>;
}



#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DefaultCollate;


impl Collate<Tensor1D<{ M }>> for DefaultCollate
{
    type Output<const B: usize> = Tensor2D<B, { M }>;

    fn collate(&self, batch: [Tensor1D<{ M }>; B]) -> Self::Output<{ B }> {


    }
}