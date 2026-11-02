"""Helper that sets replication-stress parameters."""
from ..logger import log

def apply_stress(args):
    if args.REP_WITH_STRESS:
        args.set("REP_T_STD_FACTOR", 0.2); args.set("REP_SPEED_SCALE", 1.0); args.set("REP_INIT_RATE_SCALE", 2.0)
        log.success("REP_WITH_STRESS=True -> REP_T_STD_FACTOR=0.2, REP_SPEED_SCALE=1.0, REP_INIT_RATE_SCALE=2.0")
