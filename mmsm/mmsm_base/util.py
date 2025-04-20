"""A collection of utility functions used by HierarchicalMSM, and related classes"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

import time
from collections import defaultdict
import scipy
import numpy as np
from queue import PriorityQueue


class max_fractional_difference_update_condition:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, vertex):
        diff =  max_fractional_difference(vertex._last_update_sent, vertex.get_external_T())
        return diff >= self.threshold


class UniquePriorityQueue:
    """
    UniquePriorityQueue

    A class for a simple priority queue, with no duplicate values (even if they have different
    priorities).
    """
    def __init__(self):
        self._queue = PriorityQueue()
        self._items = set()

    def not_empty(self):
        """
        Returns True iff the queue is not empty
        """
        return len(self._items) > 0

    def put(self, item):
        """
        Add an item to the queue. If the item is already in the queue, will not be added.

        Parameters
        ----------
        item : tuple
            A tuple (priority, value)
        """
        if item in self._items:
            return item
        self._items.add(item)
        self._queue.put(item)
        return item

    def get(self):
        """
        Get a value from the queue with minimal priority. If the queue is empty returns None.
        """
        if self.not_empty():
            item = self._queue.get()
            self._items.remove(item)
            return item
        return None

    def peek(self):
        if self.not_empty():
            priority, item = self._queue.queue[0]
            return priority, item
        return None

def count_dict(depth=1):
    depth = int(depth)
    if depth==1:
        return defaultdict(int)
    elif depth==2:
        return defaultdict(count_dict)
    else:
        raise ValueError("count_dict only supports depths 1 and 2.")


get_unique_id = lambda : np.random.choice(2**31)


def get_threshold_check_function(greater_than : dict={}, less_than : dict={}, max_time=None):
    """
    Get a function that returns True if keyword arguments are greater or less than certain values,
    or if a certain amount of time has passed since calling this function

    Parameters
    ----------
    greater_than : dict, optional
        A dictionary of maximum allowed values by keyword. If the returned function is called with a
        keyword in this dictionary, the function will return True if the value is greater than the
        value assigned to that keyword in greater_than.
    less_than : dict, optional
        A dictionary of minimum allowed values by keyword. If the returned function is called with a
        keyword in this dictionary, the function will return True if the value is less than the
        value assigned to that keyword in less_than.
    max_time : int or float, optional
        The maximum time in seconds before the returned function will always be evaluated to True

    Returns
    -------
    check_function : callable
        A function that accepts keyword arguments, and returns True if one or more of the arguments
        has a value greater than the value of that keyword in greater_than or less the the value of
        that keyword in less_than, or if longer than max_time seconds have passed since this
        function was called.


    Examples
    --------
    get a function that checks whether a>5 or b>1.2:
    >>> gt = {'a' : 5, 'b' : 1.2}
    >>> check = get_threshold_check_function(gt)
    >>> check(a=3)
    False
    >>> check(a=3, b=2)
    True
    >>> check()
    False
    >>> check(undefined_keyword=0)
    False
    >>> check(undefined_keywords_have_no_effect=0, b=2)
    True

    get a function that checks whether a>5, b<0 or 3.5 seconds have passed
    >>> gt = {'a' : 5}
    >>> lt = {'b' : 1.2}
    >>> check = get_threshold_check_function(less_than=lt, greater_than=gt, max_time=3.5)
    >>> check(a=3, b=3)
    True
    >>> check(a=3, b=0)
    False
    >>> sleep(3.5)
    >>> check(any_keyword_will_work=3)
    True
    >>> check()
    True
    """
    gt = defaultdict(lambda : np.inf)
    gt.update(greater_than) # val > gt[key] will be evaluated to False if key is not in greater_than
    lt = defaultdict(lambda : - np.inf)
    lt.update(less_than) # val < lt[key] will be evaluated to False if key is not in less_than
    if max_time:
        start_time = time.time()

    def _check_function(**kwargs):
        for key, val in kwargs.items():
            if max_time is not np.inf:
                return time.time() > start_time + max_time
            if (val >= gt[key]) or (val <= lt[key]):
                return True
        return False
    return _check_function

def sparse_matrix_from_count_dict(counts, ids):
    data = []
    indices = []
    indptr = []
    id_2_inx = {id:i for i, id in enumerate(ids)}
    for src in ids:
        indptr.append(len(data))
        for dest, count in counts[src].items():
            indices.append(id_2_inx[dest])
            data.append(count)
    indptr.append(len(data))
    return scipy.sparse.csr_matrix((data, indices, indptr))


def get_parent_update_condition(condition, threshold=0.1):
    if condition in ("auto", 'fractional_difference'):
        return max_fractional_difference_update_condition(threshold)
    raise NotImplementedError(f"Parent update condition {condition} not implemented.")


def max_fractional_difference(ext_T1, ext_T2):
    """
    Get the maximal change in transition probabilities between ext_T1 and ext_T2, where ext_T1/2
    are tuples of (ids, transition_probabilities), as returned by
    HierarchicalMSMVertex.get_external_T.

    Parameters
    ----------
    ext_T1 : tuple
        the previous value of external_T
    ext_T2 : tuple
        the current value of external_T

    Returns
    -------
    the maximal difference in transition probabilities, as a fractional difference, that is:
    max[abs(1-(p1/p2)) for p1 in ext_T1 and p2 in ext_T2 with matching ids]
    """
    ids1, T1 = ext_T1
    ids2, T2 = ext_T2
    assert T1.shape == T2.shape, "external_T has changed shape since last update"
    sort1 = np.argsort(ids1) # sort by ids
    sort2 = np.argsort(ids2) # sort by ids
    max_diff = 0

    for i in range(len(sort1)):
        p1 = T1[sort1[i]]
        p2 = T2[sort2[i]]
        if p2 == 0:
            continue
        diff = np.abs(1-(p1/p2))
        max_diff = max(max_diff, diff)

    return max_diff
