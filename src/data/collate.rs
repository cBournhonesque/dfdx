use std::vec::Vec;


// Collate takes a group of individual samples and builds a batch from them
pub trait Collate<T>: Default + Clone {
    // type of collate function's output = type of a batch
    type Output;

    fn collate(&self, samples: Vec<T>) -> Self::Output;
}



pub struct DefaultCollate;

// impl<T: HasArrayData> Collate<T> for DefaultCollate {
//     type Output = Vec<T>;
//
//     fn collate(samples: Vec<T>) -> Self::Output {
//         todo!()
//     }
// }