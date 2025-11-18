# SNOWMAN

### Building
 - Scripts will do this for you
 - Manually: make (also "make instr" for instrumented build with Score-P and "make instr-only" for a more bare-bone instrumented version)

### Running
 - timing.sh: Runs strong and weak scaling study and Linaroforge as in exercise 01
 - reduced-timings.sh: Runs smaller strong and weak scaling study. No Linaroforge.
 - instrument.sh: Runs Score-P. First time: will do profiling run first. This can be viewed with "CubeGui". Only counts, no timeline. This is used to compute a filter (and theoretically expected memory usage, but currently we just always set memory to 512mb). Then we do a tracing run. The output can be viewed in Vampir.

"timings.sh" and "reduced-timings.sh" can have multiple versions running in parallel, since they copy the binary to the working directory. "instrument.sh" cannot have different versions running in parallel.

### Plotting data
 - For timing data: use "plot-reduced.py" to generate plots
 - As I said, we can run "instrument.sh" and then use CubeGui and Vampir

### Results
 - results1: What the first performance test seems to suggest: the worker size should be either all / (#threads * 8) or all / (#threads * 16) and the root process should have something between 1/4 and 1/16 of that tilesize.

   We should retest that and also add 1/8 as an options. These will yield 6 different combinations.

   Note that the "failed" runs happened when SLURM retired the worker thread.
