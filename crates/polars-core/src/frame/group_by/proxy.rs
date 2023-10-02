use std::mem::ManuallyDrop;
use std::ops::Deref;

use polars_arrow::utils::CustomIterTools;
use rayon::iter::plumbing::UnindexedConsumer;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::{slice_slice, NoNull};
use crate::POOL;

/// Indexes of the groups, the first index is stored separately.
/// this make sorting fast.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GroupsIdx {
    // Invariants: first.len() + 1 == indexes.len()
    pub(crate) sorted: bool,
    first: Vec<IdxSize>,
    all: Vec<IdxSize>,
    indexes: Vec<IdxSize>,
}

pub type IdxItem = (IdxSize, Vec<IdxSize>);
pub type BorrowIdxItem<'a> = (IdxSize, &'a [IdxSize]);

impl From<Vec<IdxItem>> for GroupsIdx {
    fn from(v: Vec<IdxItem>) -> Self {
        v.into_iter().collect()
    }
}

mod internal {
    use super::IdxSize;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum VerifyGroupsIdx {
        Ok,
        LengthErr,
        MonotonicErr,
        IndexErr,
    }

    pub(crate) fn verify_groups_idx(
        first: &Vec<IdxSize>,
        all: &Vec<IdxSize>,
        indexes: &Vec<IdxSize>,
    ) -> VerifyGroupsIdx {
        // Length check
        if first.len() + 1 != indexes.len() {
            return VerifyGroupsIdx::LengthErr;
        }

        // indexes must be strictly monotonically increasing
        if !indexes.windows(2).all(|w| w[0] < w[1]) {
            return VerifyGroupsIdx::MonotonicErr;
        }

        // Last index must not be greater than the length of all if groups is not empty
        if all.len() > 0 && *indexes.last().unwrap() > all.len() as IdxSize {
            return VerifyGroupsIdx::IndexErr;
        }

        VerifyGroupsIdx::Ok
    }
}

use internal::{verify_groups_idx, VerifyGroupsIdx};

impl From<Vec<(Vec<IdxSize>, Vec<Vec<IdxSize>>)>> for GroupsIdx {
    fn from(v: Vec<(Vec<IdxSize>, Vec<Vec<IdxSize>>)>) -> Self {
        // we have got the hash tables so we can determine the final
        let (first_cap, all_cap) = v.iter().fold((0_usize, 0_usize), |acc, elem| {
            let (first_cap, all_cap) = acc;
            let (first, all) = elem;
            let first_cap = first_cap + first.len();
            let all_cap = all_cap + all.iter().map(|v| v.len()).sum::<usize>();
            (first_cap, all_cap)
        });

        let mut first = Vec::with_capacity(first_cap);
        let mut indexes = Vec::with_capacity(first_cap + 1);
        indexes.push(0 as IdxSize);
        let mut all = Vec::with_capacity(all_cap);

        for (first_vals, all_vals) in v {
            first.extend(first_vals);

            for all_val in all_vals {
                all.extend(all_val.iter());
                let curr_idx = indexes.last().unwrap().clone() as IdxSize;
                let new_idx = curr_idx + all_val.len() as IdxSize;
                indexes.push(new_idx);
            }
        }

        debug_assert_eq!(
            verify_groups_idx(&first, &all, &indexes),
            VerifyGroupsIdx::Ok
        );

        GroupsIdx {
            sorted: false,
            first,
            all,
            indexes,
        }
    }
}

impl From<Vec<Vec<IdxItem>>> for GroupsIdx {
    fn from(v: Vec<Vec<IdxItem>>) -> Self {
        let (first_cap, all_cap) =
            v.iter()
                .fold((0_usize, 0_usize), |(first_cap, all_cap), elem| {
                    let first_cap = first_cap + elem.len();
                    let all_cap = all_cap + elem.iter().map(|v| v.1.len()).sum::<usize>();
                    (first_cap, all_cap)
                });

        let mut first = Vec::with_capacity(first_cap);
        let mut indexes = Vec::with_capacity(first_cap + 1);
        indexes.push(0 as IdxSize);
        let mut all = Vec::with_capacity(all_cap);

        for elem in v {
            for (first_val, all_vals) in elem {
                first.push(first_val);
                all.extend(all_vals);
                indexes.push(all.len() as IdxSize);
            }
        }

        debug_assert_eq!(
            verify_groups_idx(&first, &all, &indexes),
            VerifyGroupsIdx::Ok
        );

        GroupsIdx {
            sorted: false,
            first,
            all,
            indexes,
        }
    }
}

impl GroupsIdx {
    pub fn new(
        first: Vec<IdxSize>,
        all: Vec<IdxSize>,
        indexes: Vec<IdxSize>,
        sorted: bool,
    ) -> Self {
        debug_assert_eq!(
            verify_groups_idx(&first, &all, &indexes),
            VerifyGroupsIdx::Ok
        );

        Self {
            sorted,
            first,
            all,
            indexes,
        }
    }

    pub fn sort(&mut self) {
        let mut idx = 0;
        let first = std::mem::take(&mut self.first);
        // store index and values so that we can sort those
        let mut idx_vals = first
            .into_iter()
            .map(|v| {
                let out = [idx, v];
                idx += 1;
                out
            })
            .collect_trusted::<Vec<_>>();
        idx_vals.sort_unstable_by_key(|v| v[1]);

        let take_first = || idx_vals.iter().map(|v| v[1]).collect_trusted::<Vec<_>>();
        let take_all = || {
            let all = Vec::<IdxSize>::with_capacity(self.all.len());
            let indexes = {
                let mut v = Vec::with_capacity(self.indexes.len());
                v.push(self.indexes[0]);
                v
            };

            let (all, indexes) = idx_vals.iter().map(|[old_idx, _]| *old_idx).fold(
                (all, indexes),
                |(mut all, mut indexes), old_idx| {
                    let old_idx = old_idx as usize;
                    let old_idx_start = self.indexes[old_idx] as usize;
                    let old_idx_end = self.indexes[old_idx + 1] as usize;

                    all.extend(self.all[old_idx_start..old_idx_end].iter());
                    indexes.push(all.len() as IdxSize);

                    (all, indexes)
                },
            );

            (all, indexes)
        };

        let (first, (all, indexes)) = POOL.install(|| rayon::join(take_first, take_all));
        self.first = first;
        self.all = all;
        self.indexes = indexes;
        self.sorted = true;

        debug_assert_eq!(
            verify_groups_idx(&self.first, &self.all, &self.indexes),
            VerifyGroupsIdx::Ok
        );
    }

    pub fn is_sorted_flag(&self) -> bool {
        self.sorted
    }

    pub fn iter(&self) -> groupsidx_iter::BorrowedIter<'_> {
        self.into_iter()
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &[IdxSize]> {
        self.iter().map(|v| v.1)
    }

    pub fn sliced_iter(
        &self,
        start_idx: usize,
        end_idx: usize,
    ) -> groupsidx_iter::BorrowedIter<'_> {
        groupsidx_iter::BorrowedIter {
            orig: self,
            start_idx,
            end_idx,
        }
    }

    pub fn indexes(&self) -> &Vec<IdxSize> {
        &self.indexes
    }

    pub fn all(&self) -> &Vec<IdxSize> {
        &self.all
    }

    pub fn first(&self) -> &[IdxSize] {
        &self.first
    }

    pub fn first_mut(&mut self) -> &mut Vec<IdxSize> {
        &mut self.first
    }

    pub(crate) fn len(&self) -> usize {
        self.first.len()
    }

    pub(crate) unsafe fn get_unchecked(&self, index: usize) -> BorrowIdxItem {
        let first = *self.first.get_unchecked(index);
        let begin = *self.indexes.get_unchecked(index) as usize;
        let end = *self.indexes.get_unchecked(index + 1) as usize;
        let all = &self.all[begin..end];
        (first, all)
    }
}

impl FromIterator<IdxItem> for GroupsIdx {
    fn from_iter<T: IntoIterator<Item = IdxItem>>(iter: T) -> Self {
        let result = GroupsIdx {
            sorted: false,
            first: Vec::new(),
            all: Vec::new(),
            indexes: vec![0 as IdxSize],
        };

        iter.into_iter().fold(result, |mut res, v| {
            res.first.push(v.0);
            res.all.extend(v.1.iter());
            res.indexes.push(res.all.len() as IdxSize);
            return res;
        })
    }
}

pub mod groupsidx_iter {
    use polars_arrow::trusted_len::TrustedLen;

    use super::{BorrowIdxItem, GroupsIdx, IdxItem};

    #[derive(Debug)]
    pub struct BorrowedIter<'a> {
        pub(crate) orig: &'a GroupsIdx,
        pub(crate) start_idx: usize,
        pub(crate) end_idx: usize,
    }

    impl<'a> Iterator for BorrowedIter<'a> {
        type Item = BorrowIdxItem<'a>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.start_idx >= self.end_idx.min(self.orig.len()) {
                return None;
            }

            // Safety: start_idx is checked above
            let result = unsafe { self.orig.get_unchecked(self.start_idx) };
            self.start_idx += 1;
            Some(result)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.len();
            (len, Some(len))
        }
    }

    unsafe impl TrustedLen for BorrowedIter<'_> {}

    impl<'a> DoubleEndedIterator for BorrowedIter<'a> {
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.start_idx >= self.end_idx && self.end_idx >= self.orig.len() {
                return None;
            }

            // Safety: end_idx is checked above
            let result = unsafe { self.orig.get_unchecked(self.end_idx - 1) };
            self.end_idx -= 1;
            Some(result)
        }
    }

    impl ExactSizeIterator for BorrowedIter<'_> {
        fn len(&self) -> usize {
            self.end_idx - self.start_idx
        }
    }

    pub struct OwnedIter {
        pub(crate) iter: GroupsIdx,
        pub(crate) curr_idx: usize,
    }

    impl Iterator for OwnedIter {
        type Item = IdxItem;

        fn next(&mut self) -> Option<Self::Item> {
            if self.curr_idx >= self.iter.len() {
                return None;
            }

            // Safety: curr_idx is checked above
            let (first, all) = unsafe { self.iter.get_unchecked(self.curr_idx) };
            let all = all.to_vec();
            self.curr_idx += 1;
            Some((first, all))
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let exact_len = self.len();
            (exact_len, Some(exact_len))
        }
    }

    unsafe impl TrustedLen for OwnedIter {}

    impl ExactSizeIterator for OwnedIter {
        fn len(&self) -> usize {
            self.iter.len()
        }
    }
}

impl<'a> IntoIterator for &'a GroupsIdx {
    type Item = BorrowIdxItem<'a>;
    type IntoIter = groupsidx_iter::BorrowedIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            orig: &self,
            start_idx: 0,
            end_idx: self.len(),
        }
    }
}

impl IntoIterator for GroupsIdx {
    type Item = IdxItem;
    type IntoIter = groupsidx_iter::OwnedIter;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            iter: self,
            curr_idx: 0,
        }
    }
}

impl FromParallelIterator<IdxItem> for GroupsIdx {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = IdxItem>,
    {
        par_iter.into_par_iter().collect::<Vec<_>>().into()
    }
}

pub mod groupsidx_par_iter {
    use rayon::iter::plumbing::{
        bridge_producer_consumer, Consumer, Producer, ProducerCallback, UnindexedConsumer,
    };
    use rayon::prelude::{IndexedParallelIterator, ParallelIterator};

    use super::{groupsidx_iter, BorrowIdxItem, GroupsIdx};

    pub struct BorrowedProducer<'a> {
        iter: groupsidx_iter::BorrowedIter<'a>,
    }

    impl<'a> Producer for BorrowedProducer<'a> {
        type Item = BorrowIdxItem<'a>;
        type IntoIter = groupsidx_iter::BorrowedIter<'a>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }

        fn split_at(self, index: usize) -> (Self, Self) {
            let split_idx = std::cmp::min(self.iter.start_idx + index, self.iter.end_idx);
            let left = groupsidx_iter::BorrowedIter {
                orig: self.iter.orig,
                start_idx: self.iter.start_idx,
                end_idx: split_idx,
            };
            let right = groupsidx_iter::BorrowedIter {
                orig: self.iter.orig,
                start_idx: split_idx,
                end_idx: self.iter.end_idx,
            };
            (Self { iter: left }, Self { iter: right })
        }
    }

    pub struct BorrowedParIter<'a> {
        iter: groupsidx_iter::BorrowedIter<'a>,
    }

    impl<'a> BorrowedParIter<'a> {
        pub(crate) fn from_orig(orig: &'a GroupsIdx) -> Self {
            Self {
                iter: orig.into_iter(),
            }
        }
    }

    impl<'a> ParallelIterator for BorrowedParIter<'a> {
        type Item = BorrowIdxItem<'a>;

        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            let number_of_items = self.iter.len();
            let producer = BorrowedProducer { iter: self.iter };
            bridge_producer_consumer(number_of_items, producer, consumer)
        }
    }

    impl<'a> IndexedParallelIterator for BorrowedParIter<'a> {
        fn len(&self) -> usize {
            self.iter.len()
        }

        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            let number_of_items = self.iter.len();
            let producer = BorrowedProducer { iter: self.iter };
            bridge_producer_consumer(number_of_items, producer, consumer)
        }

        fn with_producer<CB>(self, callback: CB) -> CB::Output
        where
            CB: ProducerCallback<Self::Item>,
        {
            callback.callback(BorrowedProducer { iter: self.iter })
        }
    }
}

impl<'a> IntoParallelIterator for &'a GroupsIdx {
    type Iter = groupsidx_par_iter::BorrowedParIter<'a>;
    type Item = BorrowIdxItem<'a>;

    fn into_par_iter(self) -> Self::Iter {
        Self::Iter::from_orig(self)
    }
}

impl IntoParallelIterator for GroupsIdx {
    type Iter = rayon::vec::IntoIter<IdxItem>;
    type Item = IdxItem;

    fn into_par_iter(self) -> Self::Iter {
        self.into_iter().collect::<Vec<_>>().into_par_iter()
    }
}

/// Every group is indicated by an array where the
///  - first value is an index to the start of the group
///  - second value is the length of the group
/// Only used when group values are stored together
///
/// This type should have the invariant that it is always sorted in ascending order.
pub type GroupsSlice = Vec<[IdxSize; 2]>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupsProxy {
    Idx(GroupsIdx),
    /// Slice is always sorted in ascending order.
    Slice {
        // the groups slices
        groups: GroupsSlice,
        // indicates if we do a rolling group_by
        rolling: bool,
    },
}

impl Default for GroupsProxy {
    fn default() -> Self {
        GroupsProxy::Idx(GroupsIdx::default())
    }
}

impl GroupsProxy {
    pub fn into_idx(self) -> GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice { groups, .. } => {
                polars_warn!("Had to reallocate groups, missed an optimization opportunity. Please open an issue.");
                let result: GroupsIdx = groups
                    .iter()
                    .map(|&[first, len]| (first, (first..first + len).collect_trusted::<Vec<_>>()))
                    .collect();
                debug_assert_eq!(
                    verify_groups_idx(&result.first, &result.all, &result.indexes),
                    VerifyGroupsIdx::Ok
                );
                result
            },
        }
    }

    pub fn iter(&self) -> GroupsProxyIter {
        GroupsProxyIter::new(self)
    }

    pub fn sort(&mut self) {
        match self {
            GroupsProxy::Idx(groups) => {
                if !groups.is_sorted_flag() {
                    groups.sort()
                }
            },
            GroupsProxy::Slice { .. } => {
                // invariant of the type
            },
        }
    }

    pub(crate) fn is_sorted_flag(&self) -> bool {
        match self {
            GroupsProxy::Idx(groups) => groups.is_sorted_flag(),
            GroupsProxy::Slice { .. } => true,
        }
    }

    pub fn group_lengths(&self, name: &str) -> IdxCa {
        let ca: NoNull<IdxCa> = match self {
            GroupsProxy::Idx(groups) => groups
                .iter()
                .map(|(_, groups)| groups.len() as IdxSize)
                .collect_trusted(),
            GroupsProxy::Slice { groups, .. } => groups.iter().map(|g| g[1]).collect_trusted(),
        };
        let mut ca = ca.into_inner();
        ca.rename(name);
        ca
    }

    pub fn take_group_firsts(self) -> Vec<IdxSize> {
        match self {
            GroupsProxy::Idx(mut groups) => std::mem::take(&mut groups.first),
            GroupsProxy::Slice { groups, .. } => {
                groups.into_iter().map(|[first, _len]| first).collect()
            },
        }
    }

    pub fn take_group_lasts(self) -> Vec<IdxSize> {
        match self {
            GroupsProxy::Idx(groups) => {
                if groups.len() <= 1 {
                    Vec::default()
                } else {
                    let indexes = groups.indexes();
                    let idx_len = indexes.len();

                    // Safety: Bounds are checked
                    // groups.len() > 1 ==> groups.indexes().len() > 2
                    // ==> groups.indexes has at least 2 elements
                    let start_idx = *unsafe { indexes.get_unchecked(idx_len - 2) } as usize;
                    let end_idx = *unsafe { indexes.get_unchecked(idx_len - 1) } as usize;

                    groups.all[start_idx..end_idx].to_vec()
                }
            },
            GroupsProxy::Slice { groups, .. } => groups
                .into_iter()
                .map(|[first, len]| first + len - 1)
                .collect(),
        }
    }

    pub fn par_iter(&self) -> GroupsProxyParIter {
        GroupsProxyParIter::new(self)
    }

    /// Get a reference to the `GroupsIdx`.
    ///
    /// # Panic
    ///
    /// panics if the groups are a slice.
    pub fn unwrap_idx(&self) -> &GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice { .. } => panic!("groups are slices not index"),
        }
    }

    /// Get a reference to the `GroupsSlice`.
    ///
    /// # Panic
    ///
    /// panics if the groups are an idx.
    pub fn unwrap_slice(&self) -> &GroupsSlice {
        match self {
            GroupsProxy::Slice { groups, .. } => groups,
            GroupsProxy::Idx(_) => panic!("groups are index not slices"),
        }
    }

    pub fn get(&self, index: usize) -> GroupsIndicator {
        match self {
            GroupsProxy::Idx(groups) => {
                let first = groups.first()[index];
                let start_idx = groups.indexes()[index] as usize;
                let end_idx = groups.indexes()[index + 1] as usize;
                let all = &groups.all()[start_idx..end_idx];
                GroupsIndicator::Idx((first, all))
            },
            GroupsProxy::Slice { groups, .. } => GroupsIndicator::Slice(groups[index]),
        }
    }

    /// Get a mutable reference to the `GroupsIdx`.
    ///
    /// # Panic
    ///
    /// panics if the groups are a slice.
    pub fn idx_mut(&mut self) -> &mut GroupsIdx {
        match self {
            GroupsProxy::Idx(groups) => groups,
            GroupsProxy::Slice { .. } => panic!("groups are slices not index"),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            GroupsProxy::Idx(groups) => groups.len(),
            GroupsProxy::Slice { groups, .. } => groups.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn group_count(&self) -> IdxCa {
        match self {
            GroupsProxy::Idx(groups) => {
                let ca: NoNull<IdxCa> = groups
                    .iter()
                    .map(|(_first, idx)| idx.len() as IdxSize)
                    .collect_trusted();
                ca.into_inner()
            },
            GroupsProxy::Slice { groups, .. } => {
                let ca: NoNull<IdxCa> = groups.iter().map(|[_first, len]| *len).collect_trusted();
                ca.into_inner()
            },
        }
    }
    pub fn as_list_chunked(&self) -> ListChunked {
        match self {
            GroupsProxy::Idx(groups) => groups
                .iter()
                .map(|(_first, idx)| {
                    let ca: NoNull<IdxCa> = idx.iter().map(|&v| v as IdxSize).collect();
                    ca.into_inner().into_series()
                })
                .collect_trusted(),
            GroupsProxy::Slice { groups, .. } => groups
                .iter()
                .map(|&[first, len]| {
                    let ca: NoNull<IdxCa> = (first..first + len).collect_trusted();
                    ca.into_inner().into_series()
                })
                .collect_trusted(),
        }
    }

    pub fn unroll(self) -> GroupsProxy {
        match self {
            GroupsProxy::Idx(_) => self,
            GroupsProxy::Slice { rolling: false, .. } => self,
            GroupsProxy::Slice { mut groups, .. } => {
                let mut offset = 0 as IdxSize;
                for g in groups.iter_mut() {
                    g[0] = offset;
                    offset += g[1];
                }
                GroupsProxy::Slice {
                    groups,
                    rolling: false,
                }
            },
        }
    }

    pub fn slice(&self, offset: i64, len: usize) -> SlicedGroups {
        // Safety:
        // we create new `Vec`s from the sliced groups. But we wrap them in ManuallyDrop
        // so that we never call drop on them.
        // These groups lifetimes are bounded to the `self`. This must remain valid
        // for the scope of the aggregation.
        let sliced = match self {
            GroupsProxy::Idx(groups) => {
                let first = unsafe {
                    let first = slice_slice(groups.first(), offset, len);
                    let ptr = first.as_ptr() as *mut _;
                    Vec::from_raw_parts(ptr, first.len(), first.len())
                };

                let all = unsafe {
                    let ptr = groups.all().as_ptr() as *mut _;
                    Vec::from_raw_parts(ptr, groups.all().len(), groups.all().len())
                };

                let indexes = unsafe {
                    let indexes = slice_slice(groups.indexes(), offset, len + 1);
                    let ptr = indexes.as_ptr() as *mut _;
                    Vec::from_raw_parts(ptr, indexes.len(), indexes.len())
                };

                ManuallyDrop::new(GroupsProxy::Idx(GroupsIdx::new(
                    first,
                    all,
                    indexes,
                    groups.is_sorted_flag(),
                )))
            },
            GroupsProxy::Slice { groups, rolling } => {
                let groups = unsafe {
                    let groups = slice_slice(groups, offset, len);
                    let ptr = groups.as_ptr() as *mut _;
                    Vec::from_raw_parts(ptr, groups.len(), groups.len())
                };

                ManuallyDrop::new(GroupsProxy::Slice {
                    groups,
                    rolling: *rolling,
                })
            },
        };

        SlicedGroups {
            sliced,
            borrowed: self,
        }
    }
}

impl From<GroupsIdx> for GroupsProxy {
    fn from(groups: GroupsIdx) -> Self {
        GroupsProxy::Idx(groups)
    }
}

pub enum GroupsIndicator<'a> {
    Idx(BorrowIdxItem<'a>),
    Slice([IdxSize; 2]),
}

impl<'a> GroupsIndicator<'a> {
    pub fn len(&self) -> usize {
        match self {
            GroupsIndicator::Idx(g) => g.1.len(),
            GroupsIndicator::Slice([_, len]) => *len as usize,
        }
    }
    pub fn first(&self) -> IdxSize {
        match self {
            GroupsIndicator::Idx(g) => g.0,
            GroupsIndicator::Slice([first, _]) => *first,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct GroupsProxyIter<'a> {
    vals: &'a GroupsProxy,
    len: usize,
    idx: usize,
}

impl<'a> GroupsProxyIter<'a> {
    fn new(vals: &'a GroupsProxy) -> Self {
        let len = vals.len();
        let idx = 0;
        GroupsProxyIter { vals, len, idx }
    }
}

impl<'a> Iterator for GroupsProxyIter<'a> {
    type Item = GroupsIndicator<'a>;

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.idx = self.idx.saturating_add(n);
        self.next()
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len {
            return None;
        }

        let out = unsafe {
            match self.vals {
                GroupsProxy::Idx(groups) => {
                    let item = groups.get_unchecked(self.idx);
                    Some(GroupsIndicator::Idx(item))
                },
                GroupsProxy::Slice { groups, .. } => {
                    Some(GroupsIndicator::Slice(*groups.get_unchecked(self.idx)))
                },
            }
        };
        self.idx += 1;
        out
    }
}

pub struct GroupsProxyParIter<'a> {
    vals: &'a GroupsProxy,
    len: usize,
}

impl<'a> GroupsProxyParIter<'a> {
    fn new(vals: &'a GroupsProxy) -> Self {
        let len = vals.len();
        GroupsProxyParIter { vals, len }
    }
}

impl<'a> ParallelIterator for GroupsProxyParIter<'a> {
    type Item = GroupsIndicator<'a>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        (0..self.len)
            .into_par_iter()
            .map(|i| unsafe {
                match self.vals {
                    GroupsProxy::Idx(groups) => GroupsIndicator::Idx(groups.get_unchecked(i)),
                    GroupsProxy::Slice { groups, .. } => {
                        GroupsIndicator::Slice(*groups.get_unchecked(i))
                    },
                }
            })
            .drive_unindexed(consumer)
    }
}

pub struct SlicedGroups<'a> {
    sliced: ManuallyDrop<GroupsProxy>,
    #[allow(dead_code)]
    // we need the lifetime to ensure the slice remains valid
    borrowed: &'a GroupsProxy,
}

impl Deref for SlicedGroups<'_> {
    type Target = GroupsProxy;

    fn deref(&self) -> &Self::Target {
        self.sliced.deref()
    }
}
