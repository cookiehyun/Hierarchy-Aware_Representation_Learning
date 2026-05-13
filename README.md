# Bachelor Thesis

Code for testing NatureLM audio embeddings on AnuraSet.

## Current status

- NatureLM setup completed on the university GPU cluster.
- Single AnuraSet audio file embedding extraction tested successfully.
- Output file-level embedding shape: `[1, 4096]`.

## Files

- `src/test_embedding.py`: Extracts a NatureLM embedding for one AnuraSet audio file.
- `scripts/run_embedding_test.sh`: Slurm script for running the embedding test on GPU.
