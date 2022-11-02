pub trait Dataset {
    type DataItem;
}


// pub trait Len {
//     /// Returns the number of elements in the collection.
//     fn len(&self) -> usize;
//
//     /// Return `true` if the collection has no element.
//     fn is_empty(&self) -> bool {
//         self.len() == 0
//     }
// }

// pub trait MapDataset : Dataset + Len + GetItem {}


pub trait IterableDataset : Dataset + IntoIterator {}
