use rand::prelude::SliceRandom;
use rand::{Rng};
use crate::data::dataset::{IterableDataset};
use crate::data::collate::Collate;
use std::vec::Vec;

// DataLoader is a wrapper around a dataset to return an iterator of batches

pub trait DataLoader : IntoIterator {}

pub trait Sampler {}


// Struct used to get an iterator of batches, via into_iter()
#[derive(Debug)]
pub struct IterableDataLoader<D, R, C>
where
    R: Rng,
{
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    rng: R,
    collate_fn: C,
}

#[derive(Debug)]
pub struct IntoIter<D: Iterator, R: Rng, C: Collate<D::Item>> {
    dataset_iter: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    rng: R,
    collate_fn: C,
}

impl<D, R, C> IntoIterator for IterableDataLoader<D, R, C>
where
    D: IterableDataset,
    R: Rng,
    C: Collate<<D as IntoIterator>::Item>,
{
    // we yield batches
    type Item = C::Output;
    type IntoIter = IntoIter<D::IntoIter, R, C>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            dataset_iter: self.dataset.into_iter(),
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
            rng: self.rng,
            collate_fn: self.collate_fn,
        }
    }
}

// Iterator of batches returned by IterableDataLoader
impl<D, R, C> Iterator for IntoIter<D, R, C>
where
    D: Iterator,
    R: Rng,
    C: Collate<D::Item>,
{
    type Item = C::Output;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = self
            .dataset_iter
            .by_ref()
            .take(self.batch_size)
            .collect::<Vec<_>>();

        if batch.is_empty() {
            return None;
        }

        if batch.len() == self.batch_size || (batch.len() != self.batch_size && !self.drop_last) {
            if self.shuffle {
                batch.shuffle(&mut self.rng);
            }
            return Some(self.collate_fn.collate(batch));
        }
        None
    }
}


#[derive(Debug)]
pub struct Iter<D: Iterator, R: Rng, C: Collate<D::Item>> {
    dataset_iter: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    rng: R,
    collate_fn: C,
}

impl<'d, D, R, C> IntoIterator for &'d IterableDataLoader<D, R, C>
    where
        D: 'd,
        &'d D: IterableDataset,
        R: Rng + Clone,
        C: Collate<<&'d D as IntoIterator>::Item>,
{
    // we yield batches
    type Item = C::Output;
    type IntoIter = Iter<<&'d D as IntoIterator>::IntoIter, R, C>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            dataset_iter: self.dataset.into_iter(),
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
            rng: self.rng.clone(),
            collate_fn: self.collate_fn.clone(),
        }
    }
}

impl<'d, D, R, C> IterableDataLoader<D, R, C>
    where
        D: 'd,
        &'d D: IterableDataset,
        R: Rng + Clone,
        C: Collate<<&'d D as IntoIterator>::Item> + Clone,
{
    /// Iterate over the dataloader without consuming the underlying dataset.
    /// As it make no sens to collate reference into a tensor, by default element are copied.
    pub fn iter(&'d self) -> Iter<<&'d D as IntoIterator>::IntoIter, R, C> {
        Iter {
            dataset_iter: self.dataset.into_iter(),
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
            rng: self.rng.clone(),
            collate_fn: self.collate_fn.clone(),
        }
    }
}


impl<D, R, C> Iterator for Iter<D, R, C>
    where
        D: Iterator,
        R: Rng,
        C: Collate<D::Item>,
{
    type Item = C::Output;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = self
            .dataset_iter
            .by_ref()
            .take(self.batch_size)
            .collect::<Vec<_>>();

        if batch.is_empty() {
            return None;
        }

        if batch.len() == self.batch_size || (batch.len() != self.batch_size && !self.drop_last) {
            if self.shuffle {
                batch.shuffle(&mut self.rng);
            }
            return Some(self.collate_fn.collate(batch));
        }
        None
    }
}