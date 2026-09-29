import numpy as np

# globals populated once per worker process
_opac = None
_model = None
_star = None
_wl_highres = None
_pressure_bar = None
_species_idx = None

def init_worker(opac_path, wl_binned_path, species_names_path, base_dir):
    global _opac, _model, _star, _wl_highres, _pressure_bar, _species_idx
    import pickle
    from POSEIDON.core import create_star, define_model, wl_grid_constant_R
    from POSEIDON.constants import R_Sun

    with open(opac_path, "rb") as f:
        _opac = pickle.load(f)

    species_names = np.load(species_names_path).tolist()
    target_species = ["H2O", "CH4", "CO", "CO2", "NH3"]
    _species_idx = {n: species_names.index(n) for n in target_species}

    _pressure_bar = np.logspace(3, -6, 100)
    _wl_highres = wl_grid_constant_R(0.4, 10.0, 15000)
    _model = define_model("disequil_model", ["H2", "He"], target_species,
                          PT_profile="file_read", X_profile="file_read")
    _star = create_star(1.0 * R_Sun, 5778.0, 4.44, 0.0)


def generate_one(args):
    """args = (index, record_dict, vmr_array). Returns (index, binned_spectrum or None, success)."""
    import spectres
    from POSEIDON.core import create_planet, make_atmosphere, compute_spectrum
    from POSEIDON.constants import R_J, M_J

    i, record, vmr_array = args
    try:
        R_p = record["Rp_Rj"] * R_J
        M_p = record["Mp_Mj"] * M_J
        planet = create_planet("planet", R_p, mass=M_p, T_eq=record["Teq_K"])

        n = len(_pressure_bar)
        T_input = np.full(n, record["Teq_K"])
        X_input = np.array([
            np.full(n, 0.83), np.full(n, 0.17),
            vmr_array[_species_idx["H2O"], :], vmr_array[_species_idx["CH4"], :],
            vmr_array[_species_idx["CO"], :],  vmr_array[_species_idx["CO2"], :],
            vmr_array[_species_idx["NH3"], :],
        ])

        atmosphere = make_atmosphere(planet, _model, _pressure_bar, 10.0, R_p,
                                     T_input=T_input, X_input=X_input)
        if not atmosphere["is_physical"]:
            return i, None, False

        spec = compute_spectrum(planet, _star, _model, atmosphere, _opac,
                                _wl_highres, spectrum_type="transmission")
        if np.any(np.isnan(spec)) or np.any(np.isinf(spec)):
            return i, None, False

        wl_binned = np.load("WL_BINNED_PATH_PLACEHOLDER")
        spec_binned = spectres.spectres(wl_binned, _wl_highres, spec, verbose=False)
        return i, spec_binned, True
    except Exception:
        return i, None, False