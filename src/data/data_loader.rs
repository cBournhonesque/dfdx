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
    rng: Option<R>,
    collate_fn: C,
}


/// Basic builder for creating dataloader.
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash, Ord)]
pub struct Builder<D, R, C>
    where
        D: IterableDataset,
        R: Rng,
        C: Collate<<D as IntoIterator>::Item>,
{
    dataset: D,
    batch_size: usize,
    drop_last: bool,
    rng: Option<R>,
    collate_fn: C,
    shuffle: bool,
}

impl<D, R, C> Builder<D, R, C>
    where
        D: IterableDataset,
        R: Rng,
        C: Collate<<D as IntoIterator>::Item>,
{
    /// Create a new [`Builder`], with default fields.
    /// By default the [`Builder`] is sequential and have a `batch_size` of one.
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            batch_size: 1,
            shuffle: false,
            drop_last: false,
            rng: None,
            collate_fn: C,
        }
    }
}

impl<D, R, C> Builder<D, R, C>
    where
        D: IterableDataset,
        R: Rng,
        C: Collate<<D as IntoIterator>::Item>,
{
    /// Use a random sampler.
    pub fn shuffle(mut self) -> Builder<D, R, C> {
        self.shuffle = true;
        self
    }
    /// Set the number of elements in a batch.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Drop the lasts element if they don't feat into a batch. For instance if a dataset have 13
    /// samples and a `batch_size` of 5, the last 3 samples will be droped.
    pub fn drop_last(mut self) -> Self {
        self.drop_last = true;
        self
    }

    /// Set a custom rng object.
    pub fn rng<RF>(self, rng: RF) -> Builder<D, RF, C>
        where
            RF: Rng
    {
        Builder {
            dataset: self.dataset,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            rng: rng,
            collate_fn: self.collate_fn,
            shuffle: self.shuffle,
        }
    }

    /// Set a custom collate function.
    pub fn collate_fn<CF>(self, collate_fn: CF) -> Builder<D, R, CF>
        where
            CF: Collate<D::Item>,
    {
        Builder {
            dataset: self.dataset,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            rng: self.rng,
            collate_fn,
            shuffle: self.shuffle,
        }
    }

    /// Create a `Dataloader` from a [`Builder`].
    pub fn build(self) -> IterableDataLoader<D, R, C> {
        IterableDataLoader {
            dataset: self.dataset,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            rng: self.rng,
            collate_fn: self.collate_fn,
            shuffle: self.shuffle,
        }
    }
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