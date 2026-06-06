# NumPy Practice — 100 Questions (Sections 1–13)

> All questions must be solved using **only** the functions and topics introduced in Sections 1–13 of the notebook.  
> No external libraries (pandas, scikit-learn, etc.) and no Bonus-section functions (`np.linalg`, `np.random.default_rng`, etc.).  
> Difficulty increases steadily from Q1 (trivial) to Q100 (full pipeline challenge).

---

## Difficulty Guide

| Range | Level | Topics Focused |
|-------|-------|---------------|
| Q1–Q25 | 🟢 Easy | Array creation, attributes, basic indexing, simple math |
| Q26–Q60 | 🟡 Medium | Boolean ops, broadcasting, statistics, sorting, reshaping |
| Q61–Q85 | 🟠 Hard | NaN handling, combined pipelines, advanced indexing |
| Q86–Q100 | 🔴 Expert | Multi-step workflows, performance, full ML pipelines |

---

## 🟢 Easy — Questions 1 to 25

**Q1.**  
Create a 1D NumPy array containing integers from 0 to 9 (inclusive) using `np.arange`.  
Print its `dtype` and confirm it is `int64` (or `int32` on Windows).

---

**Q2.**  
Create a 1D array of exactly 7 evenly spaced values between 0.0 and 1.0 (inclusive) using `np.linspace`.  
Print the result and verify the spacing between consecutive elements is equal.

---

**Q3.**  
Create the following three arrays and print each one:
- A `(3, 3)` array of all zeros (`float64`)
- A `(2, 4)` array of all ones (`float64`)
- A `(3, 3)` identity matrix

---

**Q4.**  
Create a `(4, 4)` array filled with the value `7` using `np.full`.  
Then create a `(4, 4)` array of all ones and multiply it by 7 using vectorized arithmetic.  
Confirm both results are equal using `==`.

---

**Q5.**  
Given the following array:
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```
Print its `ndim`, `shape`, `size`, `dtype`, `itemsize`, and `nbytes`.  
Manually verify: `nbytes == size * itemsize`.

---

**Q6.**  
Create `arr = np.arange(10)` and extract:
- The element at index 3
- The last element using negative indexing
- The element second from the end

---

**Q7.**  
From `arr = np.arange(10)`, extract:
- Elements from index 2 to 6 (exclusive of 6)
- Every second element from the whole array
- Elements from index 7 onwards

---

**Q8.**  
Create a 2D array:
```python
arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
```
Extract:
- The element at row 1, column 2
- The entire second row
- The entire third column

---

**Q9.**  
Create `arr = np.array([1.5, 2.7, 3.9, 4.1])` with `float64` dtype.  
Convert it to `int32` using `.astype()` and print the result.  
What happened to the decimal part? Why?

---

**Q10.**  
Calculate `nbytes` for a `float64` array of shape `(100, 100)`.  
Then calculate `nbytes` for the same shape but `float32`.  
By what factor does memory differ? Confirm using actual arrays.

---

**Q11.**  
Create `arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])` and:
- Extract the last 4 elements
- Extract the first 3 elements
- Extract elements at indices `[0, 3, 6, 9]` using fancy indexing

---

**Q12.**  
Create a `(3, 4)` array of random floats using `np.random.rand`.  
Print the array, then multiply every element by 100 and convert to `int32` (vectorized, no loops).

---

**Q13.**  
Given `arr = np.array([10, 20, 30, 40, 50])`:
- Compute the sum using `np.sum`
- Compute the mean using `np.mean`
- Compute the minimum and maximum using `np.min` and `np.max`

---

**Q14.**  
Create `arr = np.arange(1, 13)` and reshape it into a `(3, 4)` matrix.  
Print the reshaped result and verify `shape == (3, 4)`.

---

**Q15.**  
Given `arr = np.array([4, 2, 7, 1, 9, 3, 8, 5])`, sort it in ascending order using `np.sort`.  
Confirm the original array is unchanged (i.e., `np.sort` returns a copy).

---

**Q16.**  
Create a 1D array of 6 zeros and a 1D array of 6 ones.  
Stack them vertically using `np.vstack`. What is the shape of the result?  
Then stack them horizontally using `np.hstack`. What is the shape now?

---

**Q17.**  
Given `arr = np.array([1, 2, 3, 4, 5])`:
- Add 10 to every element (scalar broadcast)
- Multiply every element by 3
- Compute `arr ** 2` (square every element)

No loops allowed.

---

**Q18.**  
From `arr = np.arange(1, 10)`, extract every other element starting from index 1 using step slicing.  
Expected result: `[2, 4, 6, 8]`.

---

**Q19.**  
Check if `np.array([1, np.nan, 3, np.nan, 5])` contains NaN values using `np.isnan`.  
Count how many NaN values are present (hint: `True == 1` in NumPy).

---

**Q20.**  
Create a 2D array of shape `(2, 6)` filled with `np.arange(12)`.  
Flatten it to 1D using `.flatten()`. Print the shape before and after.

---

**Q21.**  
Given `arr = np.array([5, 3, 8, 1, 9, 2, 7])`:
- Find the index of the maximum value using `np.argmax`
- Find the index of the minimum value using `np.argmin`

---

**Q22.**  
Create a boolean array by applying the condition `> 4` to:
```python
arr = np.array([1, 5, 3, 8, 2, 6, 4, 7])
```
Print the boolean array, then use it to extract only the values greater than 4.

---

**Q23.**  
Given `arr = np.array([1, 2, 3, 4, 5, 6])`, reshape it to `(2, 3)`.  
Transpose the result using `.T`. Print the shape before and after transposing.

---

**Q24.**  
Compute `np.cumsum` on `np.array([1, 2, 3, 4, 5])`.  
What does each element represent? Verify the last element equals the total sum.

---

**Q25.**  
Given `arr = np.array([10, 20, 30, 40, 50])`, use `np.searchsorted` to find the index  
where you would insert value `25` to keep the array sorted.  
Repeat for `10` and `50`.

---

## 🟡 Medium — Questions 26 to 60

**Q26.**  
Given:
```python
arr = np.array([5, 3, 8, 1, 9, 2, 7, 4, 6])
```
Use a boolean mask to:
- Extract all values greater than 4
- Extract all values that are even (hint: `arr % 2 == 0`)
- Extract values between 3 and 7 inclusive

---

**Q27.**  
Create a `(4, 3)` matrix of random integers (use `np.random.rand` scaled and converted to `int`).  
Compute:
- Column-wise sum (`axis=0`)
- Row-wise sum (`axis=1`)
- Grand total (all elements)

---

**Q28.**  
Given `scores = np.array([72, 55, 91, 38, 85, 63, 78, 44])`:
- Use `np.argsort` to get indices that would sort the array ascending
- Use those indices (fancy indexing) to produce the sorted array
- Use `np.argsort` on the result of `np.argsort` to assign ranks (1-based, 1 = smallest)

---

**Q29.**  
Given:
```python
a = np.array([1, 2, 3])
b = np.array([[10], [20], [30]])
```
Compute `a + b` using broadcasting. Print the result and its shape.  
Explain (in a comment) why broadcasting works here — what shapes are involved?

---

**Q30.**  
Given `arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0])`:
- Compute the mean **ignoring** NaN values using `np.nanmean`
- Compute the sum **ignoring** NaN values using `np.nansum`
- Replace all NaN values with the `np.nanmedian` of the array

---

**Q31.**  
Compute the **Z-score** of `np.array([10, 20, 30, 40, 50])` manually:
- Subtract the mean
- Divide by the standard deviation

Use `np.mean` and `np.std`. Verify the Z-scored array has mean ≈ 0 and std ≈ 1.

---

**Q32.**  
From `arr = np.arange(1, 26).reshape(5, 5)`:
- Extract the 3×3 sub-matrix from rows 1:4, columns 1:4
- Extract the entire first row and last column
- Extract every element at an even row index and even column index

---

**Q33.**  
Convert `np.array([200, 300, 400, 500])` to `int8` using `.astype(np.int8)`.  
What values do you get? Explain the overflow — where does each value wrap to?  
(Recall: `int8` range is -128 to 127.)

---

**Q34.**  
Given `arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])`:
- Compute cumulative sum along `axis=1` (row-wise)
- Compute cumulative product along `axis=0` (column-wise)

---

**Q35.**  
From `arr = np.array([10, -5, 3, -8, 7, -2, 6])`:
- Use `np.where(arr > 0, arr, 0)` to replace negatives with 0
- Use `np.where(arr > 0, 1, -1)` to create a sign array
- Use `np.where` to find **indices** where `arr < 0`

---

**Q36.**  
Given a `(4, 3)` matrix of random floats, append a **column of ones** on the right side using `np.hstack`.  
The ones column must have shape `(4, 1)` — use `np.ones((4, 1))`.  
Print the shape before and after.

---

**Q37.**  
Compute the **dot product** of `np.array([1, 2, 3])` and `np.array([4, 5, 6])` using `np.dot`.  
Verify by computing it manually: `1*4 + 2*5 + 3*6`.

---

**Q38.**  
From `arr = np.arange(24).reshape(2, 3, 4)`:
- Extract the element at position `[1, 2, 3]`
- Extract the entire 2D slice at index 0 along axis 0
- Extract the element at position `[0, 0, 0]`

---

**Q39.**  
Given `arr = np.array([0, np.inf, -np.inf, 5, np.nan, 3])`:
- Use `np.isfinite` to create a mask of finite-only values
- Use `np.isinf` to find `inf` elements
- Extract only the finite values using the mask

---

**Q40.**  
Create a `float32` array from `np.arange(1_000_000, dtype=np.float32)` and a `float64` version.  
Compare `nbytes` for both. Compute `.mean()` for both and check the difference. Is it significant?

---

**Q41.**  
Given `arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])`:
- Use fancy indexing to extract elements at indices `[0, 2, 5, 9]`
- Use fancy indexing to reverse the array without slicing (`arr[arr_reversed_indices]`)

---

**Q42.**  
Compute the following percentiles of `np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])`:
- 25th percentile (Q1)
- 50th percentile (median)
- 75th percentile (Q3)
- IQR = Q3 - Q1

---

**Q43.**  
From `arr = np.arange(16).reshape(4, 4)`:
- Split into 2 equal parts along `axis=0` using `np.vsplit`
- Split into 4 equal parts along `axis=1` using `np.hsplit`
- Reassemble the vsplit result using `np.vstack` and verify equality

---

**Q44.**  
Given `arr = np.array([5, 8, 3, 1, 9, 2, 7, 4, 6])`:
- Use `np.argsort` to get the indices that sort the array **descending** (hint: negate or reverse)
- Use those indices to reorder the array in descending order

---

**Q45.**  
From `arr = np.arange(12).reshape(3, 4)`:
- Use `np.expand_dims` to add a new axis at position 0 → shape `(1, 3, 4)`
- Use `np.expand_dims` to add a new axis at position 2 → shape `(3, 4, 1)`
- Use `np.squeeze` to remove any size-1 dimensions from the result

---

**Q46.**  
Compute `np.std` and `np.var` of `np.array([2, 4, 4, 4, 5, 5, 7, 9])`.  
Verify numerically that `std ** 2 == var` (use `round()` or direct equality).

---

**Q47.**  
Given `arr = np.array([1.5, 2.5, 3.5, 4.5])`:
- Check `arr.flags['C_CONTIGUOUS']`
- Transpose it — create a column vector using `.reshape(-1, 1)` and check flags again
- Use `np.ascontiguousarray` to restore C-contiguity if needed

---

**Q48.**  
From `arr = np.arange(20).reshape(4, 5)`:
- Extract every other row and every other column: `arr[::2, ::2]`
- Print the shape and values

---

**Q49.**  
Compute `np.sin` and `np.cos` for the angles `[0, 30, 45, 60, 90]` degrees:
- Convert to radians using `np.deg2rad`
- Compute `np.sin` and `np.cos` on the radians array
- Verify that `sin(90°) ≈ 1.0` and `cos(90°) ≈ 0.0`

---

**Q50.**  
Given `arr = np.array([[4, 1, 7], [2, 9, 3], [8, 5, 6]])`:
- Use `np.argmin(arr, axis=1)` to find the column index of the minimum in each row
- Use `np.argmax(arr, axis=0)` to find the row index of the maximum in each column

---

**Q51.**  
Given `arr = np.array([[1, 2], [3, 4]])`:
- Use `arr.ravel()` — modify the result and check if `arr` changes (is it a view?)
- Use `arr.flatten()` — modify the result and check if `arr` changes (is it a copy?)

---

**Q52.**  
Apply **min-max normalization** to `np.array([2, 5, 1, 8, 3, 7, 4, 6])`:
- Formula: `(x - min) / (max - min)`
- Use only `np.min`, `np.max`, and vectorized arithmetic
- Verify that the result's minimum is 0.0 and maximum is 1.0

---

**Q53.**  
Given `scores = np.array([85, 92, 78, 65, 88, 71, 95])`:
- Use `np.argsort` twice to assign **ranks** (rank 1 = highest score)
- The formula is: `ranks = np.argsort(np.argsort(-scores)) + 1`
- Print scores alongside their ranks

---

**Q54.**  
From `arr = np.arange(1, 13).reshape(3, 4)`:
- Transpose it using `.T` — print shape
- Use `np.swapaxes(arr, 0, 1)` — confirm it gives same result as `.T`
- Reshape the transposed array to `(12,)` then back to `(4, 3)`

---

**Q55.**  
Given `arr = np.array([1, 2, 3, 4, 5, 6])`:
- Reshape to `(2, 3)`
- Transpose to `(3, 2)`
- Flatten to `(6,)`
- Print the final values — are they in the same order as the original?

---

**Q56.**  
Given `X = np.array([[1, 2], [3, 4], [5, 6]])` (shape `(3, 2)`) and  
`W = np.array([[1, 0, -1], [0, 1, 2]])` (shape `(2, 3)`):
- Compute `Z = X @ W` — print the shape and values
- Verify element `Z[1, 2]` manually using row 1 of X and column 2 of W

---

**Q57.**  
Given `arr = np.array([7, 2, 9, 4, 1, 8, 3, 6, 5])`:
- Use `np.nonzero(arr > 5)` to get indices of values greater than 5
- Confirm this matches `np.where(arr > 5)[0]`

---

**Q58.**  
Given `a = np.array([True, False, True, True, False])` and  
`b = np.array([True, True, False, True, False])`:
- Compute logical AND using `&`
- Compute logical OR using `|`
- Compute logical NOT of `a` using `~`
- How many elements are True in each result?

---

**Q59.**  
Create a `(4, 4)` matrix `A` using `np.arange(16).reshape(4, 4)`.  
Compute its product with itself: `A @ A`.  
Manually verify element `[0, 0]` (dot product of row 0 and column 0 of A).

---

**Q60.**  
Given `data = np.array([15, 200, 18, 22, 500, 17, 19, 20, 1000, 16])`:
- Use `np.percentile` to compute Q1 and Q3
- Compute IQR = Q3 - Q1 and the lower/upper fences (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`)
- Use `np.clip` to cap outliers at the fence values
- Print values before and after capping

---

## 🟠 Hard — Questions 61 to 85

**Q61.**  
Given:
```python
temps = np.array([22.0, np.nan, 18.5, np.nan, 25.3, 20.1, np.nan, 19.8])
```
- Replace all NaN values with the **median** of the non-NaN values using `np.nanmedian` and `np.where`
- Verify no NaN values remain using `np.isnan`

---

**Q62.**  
Given `X = np.random.rand(100, 5)`:
- Compute the **column-wise Z-score** using broadcasting:  
  `(X - X.mean(axis=0)) / X.std(axis=0)`
- Verify each column of the result has mean ≈ 0 and std ≈ 1  
  (use `np.abs(mean) < 1e-10` for the check)

---

**Q63.**  
Given a `(6, 4)` matrix created from `np.arange(24).reshape(6, 4)`:
- Split it into **3 equal parts** along `axis=0` using `np.vsplit`
- Add 100 to only the **second** part
- Reassemble using `np.vstack`
- Verify the first and third parts are unchanged, second part is shifted by 100

---

**Q64.**  
Given `arr = np.arange(1, 26).reshape(5, 5)`:
- Extract the **main diagonal** using fancy indexing: `arr[[0,1,2,3,4], [0,1,2,3,4]]`
- Create a copy and set all diagonal elements to 0 using the same fancy index
- Verify the off-diagonal elements are unchanged

---

**Q65.**  
Given:
```python
arr = np.array([4.0, np.inf, 2.0, -np.inf, 8.0, np.nan, 1.0, np.inf])
```
In a single vectorized pipeline (no loops):
1. Replace `+inf` with `100.0` using `np.where` + `np.isinf`
2. Replace `-inf` with `-100.0`
3. Replace `nan` with `0.0` using `np.isnan`
Print the cleaned array.

---

**Q66.**  
Implement **Binary Cross-Entropy loss** from scratch:
```python
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
y_pred = np.array([0.9, 0.2, 0.8, 0.7, 0.3, 0.4, 0.85, 0.1])
```
- Clip `y_pred` to `[1e-7, 1 - 1e-7]` using `np.clip`
- Formula: `L = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))`
- Use `np.log`, `np.mean`

---

**Q67.**  
Given `arr = np.arange(1, 101, dtype=float).reshape(10, 10)`:
- Compute row-wise mean (`axis=1`) — shape should be `(10,)`
- Compute column-wise std (`axis=0`) — shape should be `(10,)`
- Compute the 90th percentile of the **entire** array
- Find the row index and column index of the global maximum using `np.argmax` and `np.unravel_index`  
  *(hint: `np.unravel_index` is just `divmod(flat_idx, n_cols)` using Python arithmetic)*

---

**Q68.**  
Given:
```python
data = np.array([[3.0, np.nan, 5.0],
                 [np.nan, 2.0, np.nan],
                 [1.0, 4.0, 6.0]])
```
- Compute `np.nanmean` along `axis=0` (column means)
- Replace each NaN with its **column's mean** using broadcasting and `np.where`
- Verify no NaN values remain

---

**Q69.**  
Given `A = np.arange(24).reshape(2, 3, 4)`:
- Use `np.swapaxes(A, 1, 2)` — print new shape
- Use `A.transpose(0, 2, 1)` — confirm same result
- From the swapped result, extract the 2D slice at index `[0]`

---

**Q70.**  
Apply **column-wise min-max scaling** to a `(5, 3)` random array using broadcasting:
- Compute `col_min = arr.min(axis=0)` and `col_max = arr.max(axis=0)`
- Scale: `(arr - col_min) / (col_max - col_min)`
- Verify: all column minimums of the result are 0.0 and all column maximums are 1.0

---

**Q71.**  
You have a mixed signal with outliers:
```python
signal = np.array([12, 15, 11, 14, 200, 13, 16, 12, -150, 15, 14, 13])
```
- Use `np.percentile` to find Q1, Q3, and IQR
- Compute the lower and upper fences (1.5 × IQR rule)
- Use a boolean mask to identify outlier positions
- Use `np.clip` to cap the outliers at the fence values

---

**Q72.**  
Demonstrate the difference between in-place and copy sorting:
- Create `arr = np.array([5, 3, 8, 1, 9])` twice as `arr1` and `arr2`
- Sort `arr1` using `arr1.sort()` (in-place) — verify original is modified
- Sort `arr2` using `sorted_arr2 = np.sort(arr2)` — verify original is unchanged
- Compare object IDs to show they are different arrays

---

**Q73.**  
Given `arr = np.arange(1, 13).reshape(3, 4)`:
- Compute `arr @ arr.T` — print shape and result *(note: `(3,4) @ (4,3) → (3,3)`)*
- Compute `arr.T @ arr` — print shape and result *(note: `(4,3) @ (3,4) → (4,4)`)*
- Verify element `[0, 1]` of `arr @ arr.T` manually

---

**Q74.**  
Given `arr = np.arange(1, 10).reshape(3, 3)`:
- Compute `arr @ arr` using the `@` operator
- Verify element `[1, 2]` of the result manually:  
  row 1 of `arr` = `[4, 5, 6]`, column 2 of `arr` = `[3, 6, 9]`  
  dot product = `4*3 + 5*6 + 6*9`

---

**Q75.**  
Given `arr = np.array([7, 2, 9, 4, 1, 8, 3, 6, 5])`:
- Create `key1 = arr % 3` (remainder when divided by 3)
- Sort primarily by `key1` ascending, break ties by `arr` ascending using `np.lexsort`
- Print the sorted result

---

**Q76.**  
Simulate a neural network forward pass:
```
Input X: shape (10, 3)  — 10 samples, 3 features
Weights W: shape (3, 5) — random floats
Bias b: shape (5,)      — zeros
```
- Compute `Z = X @ W + b` using broadcasting
- Apply sigmoid: `A = 1 / (1 + np.exp(-Z))`
- Print shapes of Z and A, and verify all values in A are in `(0, 1)`

---

**Q77.**  
Given `arr = np.arange(100, dtype=float).reshape(10, 10)`:
- Inject NaN at 10 specific positions using fancy indexing (choose positions yourself)
- Compute `np.nansum`, `np.nanmean`, `np.nanstd` of the full array
- Count remaining NaN values using `np.isnan` + `np.sum`

---

**Q78.**  
Given `arr = np.array([[5, 3, 8], [1, 9, 2], [7, 4, 6]])`:
- Use `np.sort(arr, axis=0)` to sort each **column** independently
- Use `np.argsort(arr, axis=0)` to get the sorting indices
- Reconstruct the column-sorted array using those indices and fancy indexing

---

**Q79.**  
Construct a `(6, 6)` checkerboard pattern (alternating 0s and 1s) using only:
- `np.zeros` to create the base
- Step slicing (`[::2, ::2]`, `[1::2, 1::2]`, etc.) to set 1s
No loops allowed.

---

**Q80.**  
Given `data = np.random.rand(1000)`:
- Sort the data using `np.sort`
- Define 4 buckets: `[0, 0.25)`, `[0.25, 0.50)`, `[0.50, 0.75)`, `[0.75, 1.0]`
- Use boolean masks with `&` to count elements in each bucket using `np.sum`
- Verify that all 4 counts sum to 1000

---

**Q81.**  
Given two arrays of shape `(5, 4)` called `A` and `B`:
- Compute the **element-wise** L2 distance between corresponding rows:  
  `np.sqrt(np.sum((A - B)**2, axis=1))`
- This should produce a 1D array of 5 distances
- Find which pair of rows is closest (minimum distance) using `np.argmin`

---

**Q82.**  
Given `X = np.random.rand(100, 5)` (100 samples, 5 features):
- For each column, find the **index** of the value closest to the column mean
- Use broadcasting: `np.argmin(np.abs(X - X.mean(axis=0)), axis=0)`
- Print the 5 indices and the values at those positions

---

**Q83.**  
Compare performance of float64 vs float32 for a large computation:
- Create `arr64 = np.random.rand(1000, 1000)` (float64 by default)
- Create `arr32 = arr64.astype(np.float32)`
- Time `arr64.sum()` and `arr32.sum()` using `time.perf_counter`
- Compare: `nbytes`, mean values, computation time

---

**Q84.**  
Given `arr = np.arange(9).reshape(3, 3)`:
- Flatten it, add 1 to every element in-place using `+=`
- Verify the **original** 2D array reflects the change (ravel is a view!)
- Now use `.flatten()`, add 1 in-place — verify the original is **NOT** changed

---

**Q85.**  
Given `data = np.random.rand(200, 4)`:
- Compute Q1, Q3, IQR **per column** using `axis=0`
- Create bounds: `lower = Q1 - 1.5*IQR`, `upper = Q3 + 1.5*IQR`
- Use boolean masking to keep only rows where **all 4 features** are within bounds
- Count how many rows remain (hint: `np.all(mask, axis=1)`)

---

## 🔴 Expert — Questions 86 to 100

**Q86.**  
Build a complete **sensor data cleaning pipeline** (no loops):

Given:
```python
sensor = np.array([22.0, np.inf, 18.5, -np.inf, 25.3, np.nan, 19.8, 21.0, np.nan, 200.0])
```
Step 1: Replace `+inf` and `-inf` with `np.nan` using `np.where` + `np.isinf`  
Step 2: Fill all `np.nan` with `np.nanmedian`  
Step 3: Cap values that are more than **2 standard deviations** from the mean using `np.clip`  
Print the array after each step.

---

**Q87.**  
Given `arr = np.arange(1, 25).reshape(4, 6)`:
- Split into 3 equal parts along `axis=1` using `np.hsplit` — each part has shape `(4, 2)`
- Multiply the first part by 10, leave middle unchanged, divide the last part by 10
- Reassemble using `np.hstack`
- Verify shape is `(4, 6)` and the values are correct

---

**Q88.**  
Implement the **ReLU activation function** and its derivative using only `np.where`:

Given `Z = np.random.randn(5, 5)`:
- ReLU: `A = np.where(Z > 0, Z, 0.0)`
- ReLU derivative mask: `dA = np.where(Z > 0, 1.0, 0.0)`
- Verify: `A * (A > 0)` gives the same result as ReLU (hint: cast boolean to float)

---

**Q89.**  
Given `X = np.random.rand(10, 4)` (10 samples, 4 features):
- Compute the **pairwise L2 distance** between the first sample and all others:  
  `distances = np.sqrt(np.sum((X[0] - X[1:]) ** 2, axis=1))`
- Find the index of the nearest neighbor using `np.argmin`
- Find the index of the farthest neighbor using `np.argmax`

---

**Q90.**  
Given `arr = np.arange(60).reshape(3, 4, 5)`:
- Compute sum along `axis=0` — print shape and result
- Compute mean along `axis=2` — print shape and result
- Find `argmax` along `axis=1` — print shape and result
- Verify: total elements are preserved conceptually (document shapes)

---

**Q91.**  
Build a **Z-score outlier removal pipeline** (no loops):

Given `X = np.random.rand(100, 4)`:
1. Compute column mean and std using `axis=0`
2. Compute Z-scores using broadcasting: `(X - mean) / std`
3. Create a mask for rows where **any** Z-score exceeds 2.5: `np.abs(Z) > 2.5`
4. Keep only rows where **no** column is an outlier: `~np.any(mask, axis=1)`
5. Print: number of rows removed and shape of cleaned data

---

**Q92.**  
Given `X = np.random.rand(50, 3)`:
- Compute the **covariance matrix manually**:
  1. Center: `X_c = X - X.mean(axis=0)`
  2. Covariance: `C = X_c.T @ X_c / (len(X) - 1)`
- Compare with `np.cov(X.T)` by checking `np.abs(C - np.cov(X.T)).max() < 1e-10`
- Interpret: which pair of features is most correlated?

---

**Q93.**  
Implement **one-hot encoding** and decoding using only broadcasting:
```python
labels = np.array([0, 2, 1, 0, 2, 1, 1, 0])
n_classes = 3
```
- Encode: `ohe = (labels[:, None] == np.arange(n_classes)).astype(int)`
- Decode back: `decoded = np.argmax(ohe, axis=1)`
- Verify `decoded` equals `labels`

---

**Q94.**  
Given `probs = np.random.rand(200)` (simulated predicted probabilities):
- Sort using `np.sort`
- Define 5 bins: `[0, 0.2)`, `[0.2, 0.4)`, `[0.4, 0.6)`, `[0.6, 0.8)`, `[0.8, 1.0]`
- Use `np.searchsorted` on the sorted array to find bin boundaries
- Count elements in each bin using boolean masks
- Verify total count = 200

---

**Q95.**  
Verify **matrix multiplication associativity** using only NumPy arrays and `@`:
```python
A = np.random.rand(4, 3)
B = np.random.rand(3, 5)
C = np.random.rand(5, 2)
```
- Compute `result1 = (A @ B) @ C`
- Compute `result2 = A @ (B @ C)`
- Verify they are equal: `np.abs(result1 - result2).max() < 1e-10`
- Print the max absolute difference

---

**Q96.**  
Benchmark **three approaches** to compute row-wise L2 norm of a `(500, 100)` array using `time.perf_counter`:

a. **Python loop** over rows: `[np.sqrt((row**2).sum()) for row in arr]`  
b. **Vectorized**: `np.sqrt((arr**2).sum(axis=1))`  
c. **In-place vectorized**: `tmp = arr.copy(); tmp **= 2; np.sqrt(tmp.sum(axis=1))`

Report: time for each approach and speedup of (b) over (a).

---

**Q97.**  
Given:
```python
data = np.arange(1, 101, dtype=float).reshape(10, 10)
```
1. Inject 15 NaN values at specific positions of your choice
2. Replace NaN in **each column** with that column's `np.nanmedian` using broadcasting and `np.where`
3. Apply column-wise Z-score standardization
4. Verify: all column means ≈ 0 and all column stds ≈ 1

---

**Q98.**  
Multi-step reshape and contiguity challenge starting from `arr = np.arange(120)`:

a. Reshape to `(2, 3, 4, 5)` — print shape  
b. Transpose to `(5, 4, 3, 2)` using `.transpose(3, 2, 1, 0)` — print shape  
c. Check `C_CONTIGUOUS` and `F_CONTIGUOUS` flags after each step  
d. Use `np.ascontiguousarray` to force C-order on the transposed array  
e. Reshape the C-contiguous result to `(20, 6)` — print final shape  
f. Verify: `size` remains 120 at every step

---

**Q99.**  
Implement **double normalization** (min-max then Z-score) on an ill-scaled dataset:
```python
X = np.random.rand(50, 4) * np.array([1, 100, 1000, 0.01])
```
Step 1: Min-max scale each column → all values in `[0, 1]`  
Step 2: Z-score standardize the min-max scaled data  

After Step 2:
- Are all column means exactly 0? (Yes)
- Are all column stds exactly 1? (Yes, if std > 0)
- What would happen if a column had all identical values (std = 0)? Handle this case safely.

---

**Q100.**  
**Full ML pipeline challenge** — implement everything from scratch using only Sections 1–13:

```python
X = np.random.rand(200, 5)
y = ((X[:, 0] + X[:, 1]) > 1.0).astype(int)  # binary target
```

1. **Z-score standardize** `X` column-wise using broadcasting
2. **Shuffle and split** 70/15/15 using `np.argsort(np.random.rand(200))` as shuffle index
3. **Initialize weights** `w = np.zeros(5)` and bias `b = 0.0`
4. **Forward pass** on `X_train`: `z = X_train @ w + b`, `y_hat = 1 / (1 + np.exp(-z))`
5. **BCE loss**: clip `y_hat` to `[1e-7, 1-1e-7]`, compute `-np.mean(y*log + (1-y)*log)`
6. **Gradient**: `grad_w = X_train.T @ (y_hat - y_train) / len(y_train)`
7. **One gradient descent step**: `w -= 0.1 * grad_w`
8. **Forward pass again** with updated `w`, compute new BCE loss
9. Print: initial loss, updated loss (should be slightly lower), gradient norm `np.sqrt((grad_w**2).sum())`

---

## Quick Reference — Allowed Functions by Section

| Section | Functions / Concepts You May Use |
|---------|----------------------------------|
| 1 | `np.array`, list creation, `time.perf_counter` |
| 2 | `np.zeros`, `np.ones`, `np.full`, `np.eye`, `np.arange`, `np.linspace`, `np.random.rand` |
| 3 | `.ndim`, `.shape`, `.size`, `.dtype`, `.itemsize`, `.nbytes` |
| 4 | `[]` indexing, `[start:stop:step]` slicing, negative indices, fancy indexing `[[...]]` |
| 5 | `.astype()`, dtype constants (`np.int8`, `np.float32`, etc.) |
| 6 | `>`, `<`, `==`, `>=`, `<=`, `&`, `\|`, `~`, boolean indexing, `np.where` |
| 7 | Broadcasting rules, scalar/vector/matrix arithmetic |
| 8 | `np.isnan`, `np.isinf`, `np.isfinite`, `np.nan`, `np.inf`, `np.nansum`, `np.nanmean`, `np.nanstd`, `np.nanmedian`, `np.nanmax`, `np.nanmin`, `np.clip` |
| 9 | `+`, `-`, `*`, `/`, `**`, `np.sin`, `np.cos`, `np.tan`, `np.deg2rad`, `np.exp`, `np.log`, `np.sqrt`, `np.abs`, `np.dot`, `@` |
| 10 | `np.sum`, `np.mean`, `np.min`, `np.max`, `np.median`, `np.var`, `np.std`, `np.percentile`, `axis=`, `np.cumsum`, `np.cumprod`, `np.argmin`, `np.argmax` |
| 11 | `.reshape()`, `-1` in reshape, `.flatten()`, `.ravel()`, `np.expand_dims`, `np.squeeze`, `.T`, `np.swapaxes`, `.transpose()`, `np.vstack`, `np.hstack`, `np.concatenate`, `np.vsplit`, `np.hsplit` |
| 12 | `np.sort`, `.sort()`, `np.argsort`, `np.lexsort`, `np.searchsorted`, `np.nonzero`, `np.clip` |
| 13 | In-place ops (`+=`, `-=`, `*=`), `np.ascontiguousarray`, `.flags['C_CONTIGUOUS']`, `time.perf_counter`, dtype choice, `np.add(out=)` |

---

*Happy practising — work through Q1 to Q25 first, only move forward when you can solve all questions in a tier without hints.*
