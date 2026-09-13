# COMP3029 Computer Vision Coursework – README
## Robust Quality Classification of Germinated Oil Palm Seeds under Real-World Distribution Shift


Notebook file : COMP3029_Improved_Pipeline_v4.ipynb
Platform      : Google Colab (recommended)
Python        : 3.10+
GPU           : Optional (CPU is sufficient; GPU speeds up training)


--------------------------------------------------------------------------------
1. REQUIRED FILES
--------------------------------------------------------------------------------

Before running the notebook, ensure the following files and folders are present
in your Google Drive under the path:

    My Drive/ComputerVision/

Expected folder structure:

    ComputerVision/
    │
    ├── seedsegment/                    ← Batch-1 (pre-cropped seeds)
    │   ├── train/
    │   │   ├── BadSeed/
    │   │   └── GoodSeed/
    │   └── test/
    │       ├── BadSeed/
    │       └── GoodSeed/
    │
    ├── NormalRoomLighting/             ← Batch-2 (lighting shift)
    │   ├── Set1/
    │   │   ├── Line_Good_Seeds (s1).jpg
    │   │   ├── Line_Good_Seeds (s1).xml
    │   │   └── ...
    │   ├── Set2/ ...
    │   └── ...
    │
    ├── LightBox/                       ← Batch-3 (pose shift)
    │   ├── Set1/
    │   │   ├── Line_GoodSeeds (s1).jpg
    │   │   ├── Line_GoodSeeds (s1).xml
    │   │   └── ...
    │   ├── Set2/ ...
    │   └── ...
    │
    ├── NormalRoomLight_annotation.csv  ← Batch-2 bounding box annotations
    └── LightBox_annotation.csv         ← Batch-3 bounding box annotations

NOTE: The XML files for Batch-2 and Batch-3 must sit in the same subfolder
as their corresponding .jpg images (e.g. NormalRoomLighting/Set9/).
The notebook scans for them recursively — no separate annotations folder needed.


--------------------------------------------------------------------------------
2. DEPENDENCIES
--------------------------------------------------------------------------------

All required libraries are pre-installed in Google Colab. No pip installs are
needed. The notebook uses:

    torch, torchvision         – model training and transforms
    numpy, pandas              – data handling
    matplotlib, seaborn        – plotting and visualisation
    scikit-learn               – metrics (accuracy, F1, confusion matrix)
    Pillow (PIL)               – image loading and cropping
    tqdm                       – progress bars
    xml.etree.ElementTree      – XML annotation parsing (standard library)
    glob, collections          – file discovery (standard library)


--------------------------------------------------------------------------------
3. HOW TO RUN (GOOGLE COLAB)
--------------------------------------------------------------------------------

Step 1 — Upload the notebook
    Go to https://colab.research.google.com
    Click File > Upload notebook
    Select: COMP3029_Improved_Pipeline_v4.ipynb

Step 2 — Connect to a runtime
    Click Runtime > Change runtime type
    Set Hardware accelerator to GPU (T4) if available, or leave as CPU
    Click Save, then click Connect (top right)

Step 3 — Mount Google Drive
    Run Cell 1 (the first code cell):

        from google.colab import drive
        drive.mount('/content/drive')

    A browser popup will appear — sign in and click Allow.
    Wait for "Mounted at /content/drive" to appear.

Step 4 — Check paths (IMPORTANT)
    Run the config cell (Section 1, second code cell).
    It will print OK or MISSING for each dataset folder.
    If any path shows MISSING, update ROOT_DIR at the top of that cell:

        ROOT_DIR = "/content/drive/MyDrive/ComputerVision"

    Adjust the path to match exactly where you placed the dataset in your Drive.

Step 5 — Run all cells in order
    Click Runtime > Run all
    OR press Shift+Enter to run each cell individually from top to bottom.

    Do NOT skip cells or run them out of order — later cells depend on
    variables and models defined in earlier cells.


--------------------------------------------------------------------------------
4. NOTEBOOK SECTION OVERVIEW
--------------------------------------------------------------------------------

Section 1  – Setup & Imports
             Mounts Drive, imports all libraries, sets random seed, defines
             device (CPU/GPU), and configures all file paths.

Section 2  – Data Loading & Transforms
             Defines train/eval transforms. Loads Batch-1 via ImageFolder.
             Loads Batch-2 and Batch-3 via CSV bounding box annotations.
             Explores XML annotation files and visualises annotated scenes.
             Prints dataset statistics and label distributions.

Section 3  – Baseline Model (ResNet18)
             Builds ResNet18 with ImageNet weights. Trains on Batch-1 for
             15 epochs. Saves best checkpoint to:
               /content/baseline_resnet18_best.pth
             Plots training and validation loss/accuracy curves.

Section 4  – Baseline Evaluation
             Evaluates the baseline on Batch-1, Batch-2, and Batch-3.
             Prints accuracy, precision, recall, and F1 for each batch.
             Plots confusion matrices and a cross-domain comparison table.
             Prints full per-class classification reports.

Section 5  – Systematic Error Analysis
             Collects misclassified examples from Batch-2 and Batch-3.
             Visualises misclassified seed crops (up to 6 examples each).
             Runs scene-level error visualisation using XML bounding boxes
             (green = correct, red = wrong overlaid on raw scene photos).
             Runs Grad-CAM on misclassified examples for both batches.
             Plots confidence histograms (correct vs error predictions).

Section 6  – Hypothesis Formulation
             Markdown cells only — no code to run.
             Documents the falsified rotation hypothesis and the revised
             appearance-invariant augmentation hypothesis.

Section 7  – Hypothesis-Driven Improvement
             Defines the improved augmentation pipeline (ColorJitter,
             RandomGrayscale, RandomErasing). Trains an identical ResNet18
             with the new augmentation for 15 epochs. Saves best checkpoint:
               /content/improved_model_best.pth

Section 8  – Controlled Experimental Validation
             Evaluates the improved model on all three batches.
             Prints side-by-side comparison table (baseline vs improved).
             Plots confusion matrices and F1 bar chart for both models.
             Prints delta F1 values and hypothesis verdict.
             Runs Grad-CAM on improved model errors for comparison.

Section 9  – Discussion & Reflection
             Prints the final results summary.
             Markdown cells contain the written discussion.


--------------------------------------------------------------------------------
5. EXPECTED RUNTIMES (approximate, CPU only)
--------------------------------------------------------------------------------

    Section 1–2  (setup + data loading)     :  ~1–2 minutes
    Section 3    (baseline training)         :  ~35–45 minutes
    Section 4    (baseline evaluation)       :  ~3–5 minutes
    Section 5    (error analysis + Grad-CAM) :  ~5–10 minutes
    Section 7    (improved model training)   :  ~35–45 minutes
    Section 8    (improved evaluation)       :  ~5–8 minutes

    Total estimated runtime (CPU)            :  ~90–120 minutes

    With GPU (T4), training sections reduce to approximately 3–5 minutes each.


--------------------------------------------------------------------------------
6. SAVED CHECKPOINTS
--------------------------------------------------------------------------------

Two model checkpoints are saved to /content/ during training:

    /content/baseline_resnet18_best.pth   – best baseline model weights
    /content/improved_model_best.pth      – best improved model weights

These are saved to Colab's local storage, not Google Drive. They will be lost
if the Colab session disconnects. To preserve them, copy to Drive after training:

    import shutil
    shutil.copy("/content/baseline_resnet18_best.pth",
                "/content/drive/MyDrive/ComputerVision/baseline_resnet18_best.pth")
    shutil.copy("/content/improved_model_best.pth",
                "/content/drive/MyDrive/ComputerVision/improved_model_best.pth")

If resuming a session with saved checkpoints already in Drive, update
BASELINE_SAVE and IMPROVED_SAVE in the config cell to point to Drive paths
and skip the training cells (Sections 3 and 7).


--------------------------------------------------------------------------------
7. REPRODUCIBILITY
--------------------------------------------------------------------------------

All random seeds are fixed to 42 across Python, NumPy, PyTorch, and CUDA.
The train/validation split uses a fixed generator seed (seed=42).
Results should be identical across runs on the same hardware.
Minor numerical differences may occur between CPU and GPU runs due to
floating-point non-determinism in CUDA operations.


--------------------------------------------------------------------------------
8. TROUBLESHOOTING
--------------------------------------------------------------------------------

"MISSING – check path" in Section 1
    → Check that ROOT_DIR matches your Drive folder name exactly.
      Drive paths are case-sensitive. Use the Colab file browser (left panel)
      to confirm the exact path.

"FileNotFoundError" when loading images
    → The CSV files contain absolute paths starting with:
          /content/drive/My Drive/AAR/dataset/
      The notebook remaps these automatically using ROOT_REPLACE.
      If images still cannot be found, check that BATCH2_IMG_DIR and
      BATCH3_IMG_DIR point to the correct folders.

"No XML files found"
    → Confirm that .xml files are inside the Set* subfolders
      (e.g. NormalRoomLighting/Set9/filename.xml) alongside the .jpg files.

Colab session disconnects during training
    → Reconnect, re-run Sections 1–2 to reload data, then load saved
      checkpoints from Drive (see Section 6 above) and skip retraining.

Out of memory error
    → Reduce BATCH_SIZE in the config cell from 32 to 16.

Kernel crash on Grad-CAM cells
    → Run the Grad-CAM cells individually rather than via Run All.
      Grad-CAM requires gradient computation and can be memory-intensive.


--------------------------------------------------------------------------------
  COMP3029 | University of Nottingham Malaysia Campus | Spring 2025/2026
================================================================================
