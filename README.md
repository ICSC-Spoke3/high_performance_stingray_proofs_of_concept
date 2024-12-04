# high_performance_stingray_proofs_of_concept
Small scripts to demonstrate functionality that will be implemented in Stingray

Example run to compare the performances:
```bash
for option in "" "--use-tsreader" "--cross" "--use-tsreader --cross"
do
    mpiexec -n 10 python parallel_proof_of_concept.py --method mpi `echo $option`
    python parallel_proof_of_concept.py --method multiprocessing  `echo $option`
    python parallel_proof_of_concept.py --method none `echo $option`
done
```
