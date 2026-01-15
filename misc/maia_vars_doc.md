
## We document basic information about the variables used in the ml_coupling for the maia-transformer approach.

---

## Core Type Templates

### `input_fields_pre : modelIn`
**Type:** Template parameter (strategy-dependent)  
**Purpose:** Preprocessed input data formatted for ML inference  
**Details:** Data type depends on inference strategy (e.g., `std::vector<double*>` for some backends). Contains extracted and formatted cubes ready for model consumption.

### `output_fields_post : modelOut`
**Type:** Template parameter (strategy-dependent)  
**Purpose:** ML model output before postprocessing  
**Details:** Raw inference results that need to be unpacked and written back to CFD fields.

---

## Model Configuration

### `model_path : string`
**Type:** `std::string`  
**Purpose:** Filesystem path to the ML model file  
**Details:** Used to load the trained transformer/neural network model for inference.

### `nFields : int`
**Type:** `static constexpr int`  
**Value:** `3`  
**Purpose:** Number of physical fields (U, V, W velocity components)  
**Details:** Fixed at compile time. Represents the 3D velocity vector field dimensions.

---

## Domain Decomposition (MPI)

### `nCells : vector<int>`
**Type:** `std::vector<int>` (size 3)  
**Purpose:** Local grid dimensions **including ghost layers** for this MPI rank  
**Details:** `[nCells[0], nCells[1], nCells[2]]` = [Z, Y, X] dimensions of this rank's subdomain.  
**Example:** `[52, 100, 100]` = 52×100×100 cells (includes 2 ghost layers per boundary if `nGhostLayers=1`).

### `nOffsetCells : vector<int>`
**Type:** `std::vector<int>` (size 3)  
**Purpose:** Starting position of this rank's subdomain in the **global grid**  
**Details:** Used for parallel I/O to write local data at correct global coordinates.  
**Example:** Rank 2 with offset `[100, 0, 0]` owns Z-slices starting at global Z=100.

### `nActiveCells : vector<int>`
**Type:** `std::vector<int>` (size 3)  
**Purpose:** Local grid dimensions **excluding ghost layers**  
**Formula:** `nActiveCells[i] = nCells[i] - 2 * nGhostLayers`  
**Details:** Only active cells are extracted into cubes for ML processing.  
**Example:** `nCells=[52,100,100]`, `nGhostLayers=1` → `nActiveCells=[50,98,98]`.

### `activeFieldCells : int`
**Type:** `int`  
**Purpose:** Total number of active cells (product of `nActiveCells`)  
**Formula:** `nActiveCells[0] × nActiveCells[1] × nActiveCells[2]`  
**Details:** Used for flat array indexing and memory allocation.

### `fullFieldCells : int`
**Type:** `int`  
**Purpose:** Total cells including ghosts (product of `nCells`)  
**Formula:** `nCells[0] × nCells[1] × nCells[2]`

### `nGhostLayers : int`
**Type:** `int`  
**Purpose:** Number of ghost cell layers on each boundary  
**Details:** Ghost cells are **excluded** from cube extraction. They contain boundary/halo data from neighboring MPI ranks for stencil operations.

---

## Cube Extraction (Spatial Decomposition)

### `cubeD : int`
**Type:** `int`  
**Purpose:** Cube dimension (edge length)  
**Details:** Each cube is `cubeD × cubeD × cubeD`. Common value: 8 → 8³=512 cells per cube.

### `cubeSize : int`
**Type:** `int`  
**Formula:** `cubeD * cubeD * cubeD`  
**Purpose:** Total cells per cube  
**Details:** Used for memory allocation and flattened cube indexing.

### `numCubes : int`
**Type:** `int`  
**Formula:** `zs.size() × ys.size() × xs.size()`  
**Purpose:** Total number of cubes extracted from local domain  
**Details:** Depends on `nActiveCells`, `cubeD`, and `overlap`.  
**Example:** 50×98×98 domain with cubeD=8, overlap=0 → ~7×13×13 = 1,183 cubes.

### `xs, ys, zs : vector<int>`
**Type:** `std::vector<int>`  
**Purpose:** Starting indices for cube extraction in each dimension  
**Calculation:**  
- **No overlap:** `linspace(0, nActiveCells[i] - cubeD, count)` → evenly spaced  
- **With overlap:** `get_full_indices(nActiveCells[i], cubeD - overlap)` → stride = `cubeD - overlap`  
**Details:** Defines the (x0, y0, z0) positions where cubes are extracted from the flat array.

### `concatX, concatY, concatZ : int`
**Type:** `int`  
**Formula:** `ceil(nActiveCells[i] / cubeD)`  
**Purpose:** Number of cubes along each dimension (initial estimate without overlap)  
**Details:** Actual cube counts are `xs.size()`, `ys.size()`, `zs.size()`.

### `overlap : int`
**Type:** `int`  
**Purpose:** Number of overlapping cells between adjacent cubes  
**Details:**  
- **overlap=0:** Cubes tile without touching, stride = `cubeD`  
- **overlap>0:** Cubes share borders, stride = `cubeD - overlap`  
**Example:** cubeD=8, overlap=2 → cubes start every 6 cells, sharing a 2-cell border.

### `totalElements : int64_t`
**Type:** `int64_t` (64-bit to prevent overflow)  
**Formula:** `inputSeqLen × nFields × numCubes × cubeSize`  
**Purpose:** Total data elements across all cubes, fields, and timesteps  
**Details:** Used for memory allocation and validation.

---

## Temporal Control (Inference Scheduling)

### `inputSeqLen : int`
**Type:** `int`  
**Purpose:** Number of historical timesteps fed to the transformer  
**Details:** Model requires `inputSeqLen` consecutive coupling steps before inference.  
**Example:** inputSeqLen=4 → model uses t-3, t-2, t-1, t to predict t+1.

### `inferenceInterval : int`
**Type:** `int`  
**Purpose:** Number of CFD timesteps between consecutive inferences  
**Constraint:** Must be ≥ `(inputSeqLen-1) × inputStepDistance`  
**Example:** inferenceInterval=10 → infer at steps 10, 20, 30, ...

### `inferenceIncrement : int`
**Type:** `int`  
**Purpose:** Additional offset added to next inference step  
**Details:** Allows irregular inference scheduling beyond fixed intervals.

### `nextInferenceStep : int`
**Type:** `int`  
**Purpose:** The next CFD timestep at which inference will run  
**Details:** Updated after each inference: `nextInferenceStep += inferenceInterval + inferenceIncrement`.

### `inferenceStartStep : int`
**Type:** `int`  
**Purpose:** First CFD timestep where inference begins  
**Constraint:** Must be ≥ `inputSeqLen - 1` (need history first).

### `forecastWindow : int`
**Type:** `int`  
**Purpose:** Number of future timesteps predicted by the model  
**Details:** Some models predict multiple steps ahead; this configures that behavior.

### `inputStepDistance : int`
**Type:** `int`  
**Purpose:** Temporal spacing between input sequence steps  
**Details:** If `inputStepDistance=2`, input at t-6, t-4, t-2, t (not consecutive).  
**Default:** Likely 1 for consecutive steps.

### `couplingSteps : vector<int>`
**Type:** `std::vector<int>`  
**Purpose:** List of CFD timesteps used for current input sequence  
**Calculation:** Backwards from `nextInferenceStep` by `inputStepDistance`  
**Example:** nextInferenceStep=10, inputSeqLen=4, distance=1 → [7, 8, 9, 10].

### `iter : int`
**Type:** `int`  
**Purpose:** Counter tracking preprocessing iterations  
**Details:** Increments 0→(inputSeqLen-1), then resets after inference.  
**Usage:** `iter < inputSeqLen-1` → gather data; `iter == inputSeqLen-1` → run inference.

---

## Preprocessing & I/O

### `scalingFactor : double`
**Type:** `double`  
**Purpose:** Data normalization/scaling factor  
**Details:** Applied during preprocessing to normalize input values (e.g., divide velocities by this factor).

### `hdfOutputInterval : int`
**Type:** `int`  
**Purpose:** Interval for writing debug HDF5 outputs  
**Details:** Every `hdfOutputInterval` steps, write fields to disk for validation.

### `totalTimesteps : int`
**Type:** `int`  
**Purpose:** Total number of CFD timesteps in the simulation  
**Details:** Used for scheduling and termination logic.

---

## Internal Storage (Debugging)

### `m_preFieldCubes : vector<vector<double>>`
**Type:** `std::vector<std::vector<double>>`  
**Purpose:** Stores extracted cubes **before** inference (one vector per field)  
**Structure:** `m_preFieldCubes[fieldIdx][cubeIdx * cubeSize + localIdx]`  
**Details:** Used for CSV export/debugging. Not the actual input to ML (that's `input_fields_pre`).

### `m_postFieldCubes : vector<vector<double>>`
**Type:** `std::vector<std::vector<double>>`  
**Purpose:** Stores output cubes **after** inference (one vector per field)  
**Details:** Mirrors `m_preFieldCubes` structure for output validation.

---

## Control Flags

### `firstMLStep : bool`
**Type:** `bool`  
**Initial:** `true`  
**Purpose:** Track first ML step for Score-P profiling initialization  
**Details:** Toggles to `false` after first complete inference cycle.

### `finalized : bool`
**Type:** `bool`  
**Purpose:** Track if resources have been cleaned up  
**Details:** Prevents double-free errors in destructor.

