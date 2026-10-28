# Legacy (December 2024)

The original notebooks and scripts, kept as a record of how the first results
were produced. 

Note that paths are Windows-specific, and `download_data.py` scrapes a page that the city has since moved.

## Evaluation

The the 2024 evaluation measured:

- One sample per community district: 59 districts split in half by file
  order, so the test set held about 30 point clouds.
- There was no validation set. The reported 83% was the best single epoch
  on the test set (25 of 30 correct). Test accuracy ranged between 60% and 83%
  from epoch to epoch, and the saved `models/model100.pt` scored 76.7%.
- Points were raw Brep vertices that could contain duplicates. The
  network's input features were constant, so it saw geometry only through
  offsets between neighboring points.

The `pcc` package in the repository root replaces this with district-level
tiles, a held-out validation set for checkpoint selection, and multi-seed
reporting.
