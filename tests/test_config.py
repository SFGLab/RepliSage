"""Config: defaults < ini < CLI, aliases, types, round trip."""
import pytest
from src.config.loader import get_config, read_ini
from src.config.writer import write_config

def test_defaults_load(tmp_path):
    a, _ = get_config([], ini=None)
    assert a.N_BEADS is None or isinstance(a.N_BEADS, int)

def test_cli_beats_ini(tmp_path):
    ini = tmp_path / "c.ini"; ini.write_text("[x]\nN_LEF = 7\nN_BEADS = 50\n")
    a, _ = get_config(["-c", str(ini)])
    assert a.N_LEF == 7 and a.N_BEADS == 50
    a, _ = get_config(["-c", str(ini), "--seed", "9"])
    assert a.SEED == 9 and a.N_LEF == 7 and a.N_BEADS == 50

def test_section_names_and_case_do_not_matter(tmp_path):
    ini = tmp_path / "c.ini"; ini.write_text("[whatever]\nn_lef = 11\n")
    assert get_config(["-c", str(ini)])[0].N_LEF == 11

def test_aliases(tmp_path):
    ini = tmp_path / "c.ini"; ini.write_text("[x]\nPOTTS_INTERACT_COEFF = 0.5\nPOTTS_FIELD_COEFF = 0.25\n")
    a, _ = get_config(["-c", str(ini)])
    assert a.EPI_INTERACT_COEFF == 0.5 and a.EPI_FIELD_COEFF == 0.25

def test_bool_parsing():
    a, _ = get_config([], overrides={"SAVE_PLOTS": "False", "VIZ_HEATS": "True"})
    assert a.SAVE_PLOTS is False and a.VIZ_HEATS is True

def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_config(["-c", str(tmp_path / "nope.ini")])

def test_unknown_ini_key_is_ignored(tmp_path):
    ini = tmp_path / "c.ini"; ini.write_text("[x]\nNOT_A_KEY = 1\nN_LEF = 3\n")
    assert get_config(["-c", str(ini)])[0].N_LEF == 3

def test_write_read_round_trip(tmp_path):
    a, _ = get_config(["--region", "chr2"], overrides={"N_LEF": 13})
    f = tmp_path / "out.ini"; write_config(a, str(f))
    b, _ = get_config(["-c", str(f)])
    assert b.N_LEF == 13 and b.CHROM == "chr2"


def test_main_flags():
    a, _ = get_config(["--bedpe-file", "x.bedpe", "--region", "chr14:80,835,000-98674700", "--rt-file", "t.bigWig", "--run-md", "true", "--with-stress", "--n-beads", "200", "--no-movie"])
    assert (a.BEDPE_PATH, a.CHROM, a.REGION_START, a.REGION_END) == ("x.bedpe", "chr14", 80835000, 98674700)
    assert a.REPT_PATH == "t.bigWig" and a.USE_SC_REPT is False and a.SIMULATION_TYPE == "MD" and a.N_BEADS == 200 and a.MOVIE is False
    assert a.REP_WITH_STRESS is True and a.REP_T_STD_FACTOR == 0.2            # stress preset applied

def test_flags_underscore_and_rt_kinds():
    assert get_config(["--bedpe_file", "a", "--rt_file", "s.parquet"])[0].USE_SC_REPT is True
    a, _ = get_config(["--rt-file", "builtin"]); assert a.USE_SC_REPT is True and a.REPT_PATH is None
    assert get_config([])[0].REPT_PATH is None and get_config([])[0].SIMULATION_TYPE is None     # no data, no 3D by default

def test_run_md_false_beats_file(tmp_path):
    ini = tmp_path / "c.ini"; ini.write_text("[x]\nsimulation_type = MD\n")
    assert get_config(["-c", str(ini), "--run-md", "false"])[0].SIMULATION_TYPE is None

def test_old_parameter_flags_are_gone():
    with pytest.raises(SystemExit): get_config(["--n_lef", "5"])

def test_help_lists_main_options(capsys):
    with pytest.raises(SystemExit): get_config(["-h"])
    out = capsys.readouterr().out
    for k in ("--bedpe-file", "--rt-file", "--run-md", "--with-stress", "--config", "config file"): assert k in out


def test_step_flags(tmp_path):
    """Sampling, S-phase and MD step flags reach the parameters."""
    a, _ = get_config(["--n-steps", "9000", "--mc-step", "50", "--burnin", "300", "--rep-start", "3000", "--rep-duration", "4000",
                       "--md-steps", "200", "--md-step-size", "50 femtosecond"])
    assert (a.N_STEPS, a.MC_STEP, a.BURNIN, a.REP_START_TIME, a.REP_TIME_DURATION, a.SIM_STEP) == (9000, 50, 300, 3000, 4000, 200)
    assert a.INTEGRATOR_STEP == "50 femtosecond"
