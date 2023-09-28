use std::mem::ManuallyDrop;
use std::ops::Deref;

use polars_arrow::utils::CustomIterTools;
use rayon::iter::plumbing::UnindexedConsumer;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::{flatten, slice_slice, NoNull};
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
pub type BorrowIdxItem<'a> = (IdxSize, &'a Vec<IdxSize>);

impl From<Vec<IdxItem>> for GroupsIdx {
    fn from(v: Vec<IdxItem>) -> Self {
        v.into_iter().collect()
    }
}

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
                all.extend(all_val);
                let curr_idx = indexes.last().unwrap().clone() as IdxSize;
                let new_idx = curr_idx + all_val.len() as IdxSize;
                indexes.push(new_idx);
            }
        }

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
                let curr_idx = indexes.last().unwrap().clone() as IdxSize;
                let new_idx = curr_idx + all.len() as IdxSize;
                indexes.push(new_idx);
            }
        }

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
            idx_vals
                .iter()
                .map(|v| unsafe {
                    let idx = v[0] as usize;
                    std::mem::take(self.all.get_unchecked_mut(idx))
                })
                .collect_trusted::<Vec<_>>()
        };
        let (first, all) = POOL.install(|| rayon::join(take_first, take_all));
        self.first = first;
        self.all = all;
        self.sorted = true
    }
    pub fn is_sorted_flag(&self) -> bool {
        self.sorted
    }

    pub fn iter(
        &self,
    ) -> std::iter::Zip<std::iter::Copied<std::slice::Iter<IdxSize>>, std::slice::Iter<Vec<IdxSize>>>
    {
        self.into_iter()
    }

    pub fn all(&self) -> &[Vec<IdxSize>] {
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
        let all = self.all.get_unchecked(index);
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
            let new_idx = res.indexes.last().unwrap().clone() + (v.1.len() as IdxSize);
            res.indexes.push(new_idx);
            return res;
        })
    }
}

impl<'a> IntoIterator for &'a GroupsIdx {
    type Item = BorrowIdxItem<'a>;
    type IntoIter = std::iter::Zip<
        std::iter::Copied<std::slice::Iter<'a, IdxSize>>,
        std::slice::Iter<'a, Vec<IdxSize>>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.first.iter().copied().zip(self.all.iter())
    }
}

impl IntoIterator for GroupsIdx {
    type Item = IdxItem;
    type IntoIter = std::iter::Zip<std::vec::IntoIter<IdxSize>, std::vec::IntoIter<Vec<IdxSize>>>;

    fn into_iter(mut self) -> Self::IntoIter {
        let all: Vec<Vec<IdxSize>> = self
            .first
            .iter()
            .enumerate()
            .map(|(idx, v)| unsafe {
                let start = *self.indexes.get_unchecked(idx) as usize;
                let end = *self.indexes.get_unchecked(idx + 1) as usize;
                let all_vals = std::mem::take(&mut self.all[start..end].to_vec());
                all_vals
            })
            .collect();

        self.first.into_iter().zip(all)
    }
}

impl FromParallelIterator<IdxItem> for GroupsIdx {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = IdxItem>,
    {
        let (first, all) = par_iter.into_par_iter().unzip();
        GroupsIdx {
            sorted: false,
            first,
            all,
        }
    }
}

impl<'a> IntoParallelIterator for &'a GroupsIdx {
    type Iter = rayon::iter::Zip<
        rayon::iter::Copied<rayon::slice::Iter<'a, IdxSize>>,
        rayon::slice::Iter<'a, Vec<IdxSize>>,
    >;
    type Item = BorrowIdxItem<'a>;

    fn into_par_iter(self) -> Self::Iter {
        self.first.par_iter().copied().zip(self.all.par_iter())
    }
}

impl IntoParallelIterator for GroupsIdx {
    type Iter = rayon::iter::Zip<rayon::vec::IntoIter<IdxSize>, rayon::vec::IntoIter<Vec<IdxSize>>>;
    type Item = IdxItem;

    fn into_par_iter(mut self) -> Self::Iter {
        let first = std::mem::take(&mut self.first);
        let all = std::mem::take(&mut self.all);
        first.into_par_iter().zip(all.into_par_iter())
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
                groups
                    .iter()
                    .map(|&[first, len]| (first, (first..first + len).collect_trusted::<Vec<_>>()))
                    .collect()
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
                groups
                    .all
                    .iter()
                    .map(|idx| {
                        // safety:
                        // idx has at least one eletment, so -1 is always in bounds
                        unsafe { *idx.get_unchecked(idx.len() - 1) }
                    })
                    .collect()
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
                let first = groups.first[index];
                let all = &groups.all[index];
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
                    let all = slice_slice(groups.all(), offset, len);
                    let ptr = all.as_ptr() as *mut _;
                    Vec::from_raw_parts(ptr, all.len(), all.len())
                };
                ManuallyDrop::new(GroupsProxy::Idx(GroupsIdx::new(
                    first,
                    all,
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
