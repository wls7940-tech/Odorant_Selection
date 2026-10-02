Odorant Selection
============================

This section compares the model-predicted top-5 odor descriptors
with the GT vectors in `GT_vectors.xlsx`.

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
