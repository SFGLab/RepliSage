# Demos

Short runs to try RepliSage (run from the repository root).

    replisage -c demo/demo_loops_only.ini -v          # loop extrusion only
    replisage -c demo/demo_with_replication.ini -v    # + replication (built-in GM12878 single-cell timing)
    python demo/test_inputs.py                        # checks the error handling of the inputs

`loops_demo.bedpe` has CTCF orientation (columns 8-9); `loops_no_orientation.bedpe` does not.
Results go to `demo/results/`.
