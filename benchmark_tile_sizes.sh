#!/bin/bash
#SBATCH --job-name=tile_benchmark         
#SBATCH --account=tmp_hpca_workshop
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=96              
#SBATCH --cpus-per-task=1                 
#SBATCH --output=tile_benchmark_%j.out    
#SBATCH --error=tile_benchmark_%j.err     
#SBATCH --time=01:00:00                   
#SBATCH --partition=intelsr_devel
#SBATCH --exclusive 

unset SLURM_EXPORT_ENV

module load OpenMPI/4.1.6-GCC-13.2.0

# Clean and build
make clean
make

# Tile size benchmark test (fixed problem size: 1024 4, varying tile sizes and process counts)
echo "Starting Tile Size Benchmark Tests..."

# Run tests for each tile size at process counts from 20 to 96
for tile_size in 8 16 32 48 64 128
do
    echo ""
    echo "=== Testing Tile Size: ${tile_size}x${tile_size} ==="
    
    for n in {20..96..4}
    do
        echo "Running test with tile size ${tile_size}x${tile_size} at $n processes..."
        
        time mpirun -n $n ./snowman 1024 4 $tile_size
    done
done

echo "Tile size benchmark tests completed."
