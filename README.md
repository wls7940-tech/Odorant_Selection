Odorant Selection
============================
This script is intended to be used after running the SCENT model [1]. It takes the top-5 predicted odor descriptors from SCENT and performs angular-distance-based matching against the GT odorant vectors.
This section compares the model-predicted top-5 odor descriptors with the GT vectors in `GT_vectors.xlsx`.
The final output is the odorant whose GT vector has the smallest angular distance to the prediction vector, along with the corresponding angular distance.

Before running the code, check the following items.


1. GT vector file
-----------------

Place:

    GT_vectors.xlsx

in the same directory as the Python script.

The Excel file must contain these columns:

    odorant
    descriptor
    weight


2. Input CSV directory
----------------------

In the MAIN section, find:

    INPUT_CSV_DIR = r"path/to/your/input_csv_directory"

Replace it with the directory containing your input CSV files.


3. Model checkpoint path
------------------------

In the model inference section, find:

    base_path

and set it to the directory containing your model checkpoint.

Also check:

    ckpt_template

and make sure the checkpoint filename is correct.


4. Angular distance
-------------------

Angular distance is calculated in radians:

    angular_distance = arccos(cosine_similarity)

Smaller values indicate greater similarity.


## Reference
[1] Zhang, Z. et al. SCENT: Aligning Mass Spectra with Molecular Structure for Olfactory Perception. Preprint at https://doi.org/10.48550/ARXIV.2605.27009 (2026)

[2] Good Scents Company. The Good Scents Company Information System. https://www.thegoodscentscompany.com/
