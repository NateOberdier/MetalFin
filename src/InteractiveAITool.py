from ipywidgets import Dropdown, FloatText, Button, HBox, Output, Layout
from IPython.display import display, clear_output
import numpy as np
import pandas as pd

metal_col_names = list(metal_cols.values())

absorption_raw = {
    "mercury": 0.08,
    "arsenic": 0.95,
    "cadmium": 0.05,
    "lead": 0.10,
}

absorption_cooked = {
    "mercury": 0.92,
    "arsenic": 0.95,
    "cadmium": 0.05,
    "lead": 0.10,
}

def evaluate_fish_intake(
    species,
    production,
    preparation,
    amount_value,
    amount_unit="lb",
    body_weight_kg=80.0
):
    if amount_value is None or amount_value <= 0:
        amount_value = 0.5

    if amount_unit == "lb":
        mass_g = amount_value * 453.592
        unit_label = "lb"
        mass_to_unit = 1.0 / 453.592
    elif amount_unit == "kg":
        mass_g = amount_value * 1000.0
        unit_label = "kg"
        mass_to_unit = 1.0 / 1000.0
    else:
        mass_g = amount_value
        unit_label = "g"
        mass_to_unit = 1.0

    row_cat = {}
    for c in cat_features:
        if c == "species":
            row_cat[c] = str(species)
        elif c == "production":
            row_cat[c] = str(production)
        else:
            row_cat[c] = "Unknown"

    row_num = {}
    for c in num_features:
        row_num[c] = num_feature_means.get(c, 0.0)

    row = {**row_cat, **row_num}
    X_input = pd.DataFrame([row])

    pred_max_metal = float(regression_pipeline.predict(X_input)[0])

    metal_preds = {}
    if species in species_stats.index:
        stats_row = species_stats.loc[species]
        for metal, col in metal_cols.items():
            ratio_col = ratio_cols.get(metal)
            if (
                ratio_col in stats_row.index
                and not pd.isna(stats_row[ratio_col])
                and pred_max_metal > 0
            ):
                conc = stats_row[ratio_col] * pred_max_metal
            else:
                conc = stats_row[col] if col in stats_row.index else np.nan
            metal_preds[metal] = float(conc) if not pd.isna(conc) else np.nan
    else:
        metal_preds = {metal: float("nan") for metal in metal_cols}

    metal_intake_ug = {}
    for metal, conc in metal_preds.items():
        if pd.isna(conc):
            metal_intake_ug[metal] = np.nan
        else:
            metal_intake_ug[metal] = conc * mass_g

    absorption_map = absorption_raw if preparation == "raw" else absorption_cooked
    metal_absorbed_ug = {}
    for metal, intake in metal_intake_ug.items():
        if pd.isna(intake):
            metal_absorbed_ug[metal] = np.nan
        else:
            frac = absorption_map.get(metal, 1.0)
            metal_absorbed_ug[metal] = intake * frac

    TDI_UG_PER_KG_PER_DAY = 0.1
    weekly_tolerance = TDI_UG_PER_KG_PER_DAY * body_weight_kg * 7.0

    metal_ratios = {}
    for metal, absorbed in metal_absorbed_ug.items():
        if not pd.isna(absorbed) and weekly_tolerance > 0:
            metal_ratios[metal] = absorbed / weekly_tolerance

    if metal_ratios:
        limiting_metal = max(metal_ratios, key=metal_ratios.get)
        max_ratio = metal_ratios[limiting_metal]
    else:
        limiting_metal = None
        max_ratio = 0.0

    if max_ratio <= 1:
        safety = "SAFE"
    elif max_ratio <= 2:
        safety = "RISKY"
    else:
        safety = "DANGEROUS"

    margin_info = None
    if (
        limiting_metal is not None
        and limiting_metal in metal_preds
        and not pd.isna(metal_preds[limiting_metal])
        and weekly_tolerance > 0
    ):
        conc_lim = metal_preds[limiting_metal]
        frac_lim = absorption_map.get(limiting_metal, 1.0)
        if conc_lim > 0 and frac_lim > 0:
            mass_g_safe = weekly_tolerance / (conc_lim * frac_lim)
            mass_g_risky = 2.0 * weekly_tolerance / (conc_lim * frac_lim)

            mass_safe_unit = mass_g_safe * mass_to_unit
            mass_risky_unit = mass_g_risky * mass_to_unit
            current_unit = mass_g * mass_to_unit

            margin_info = {
                "limiting_metal": limiting_metal,
                "mass_safe_unit": mass_safe_unit,
                "mass_risky_unit": mass_risky_unit,
                "current_unit": current_unit,
                "unit_label": unit_label,
            }

    lines = []
    lines.append("======================================")
    lines.append(f"Species: {species} ({production}, {preparation})")
    lines.append(f"Amount per week: {amount_value:.2f} {amount_unit} (~{mass_g:.1f} g)")
    lines.append("--------------------------------------")

    if preparation == "raw" and production == "farmed":
        lines.append("⚠️ WARNING: Farmed seafood should NOT be consumed raw due to higher contamination and pathogen risks.")
        lines.append("⚠️ Choose WILD selections instead.")
        lines.append("--------------------------------------")

    lines.append(f"Model-predicted MAX metal (mg/kg): {pred_max_metal:.4f}")
    lines.append("")
    lines.append("Estimated per-metal absorption and intake:")

    for metal in metal_cols.keys():
        conc = metal_preds.get(metal, np.nan)
        total_intake = metal_intake_ug.get(metal, np.nan)
        absorbed = metal_absorbed_ug.get(metal, np.nan)
        frac = absorption_map.get(metal, 1.0)
        metal_name = metal.capitalize()
        if pd.isna(conc) or pd.isna(total_intake) or pd.isna(absorbed):
            lines.append(f"  {metal_name:8s} | conc: N/A    | absorbed: N/A | total: N/A")
        else:
            lines.append(
                f"  {metal_name:8s} | conc ≈ {conc:7.4f} mg/kg | absorbed ≈ {absorbed:8.1f} µg"
                f" ({frac*100:4.1f}% of {total_intake:8.1f} µg)"
            )

    lines.append("")
    lines.append("Overall safety classification (based on absorbed load):")
    lines.append(f"  >>> {safety} <<<")

    if margin_info is not None:
        lm = margin_info["limiting_metal"].capitalize()
        mass_safe = margin_info["mass_safe_unit"]
        mass_risky = margin_info["mass_risky_unit"]
        current = margin_info["current_unit"]
        unit_label = margin_info["unit_label"]

        lines.append("")
        lines.append("Margin to change classification (approx):")
        lines.append(f"  (Driven mostly by {lm})")

        if safety == "SAFE":
            add_to_risky = max(mass_risky - current, 0.0)
            add_to_safe_edge = max(mass_safe - current, 0.0)
            lines.append(f"  • Add ≈ {add_to_safe_edge:.2f} {unit_label} to reach SAFE/RISKY boundary.")
            lines.append(f"  • Add ≈ {add_to_risky:.2f} {unit_label} to reach RISKY/DANGEROUS boundary.")
        elif safety == "RISKY":
            remove_to_safe = max(current - mass_safe, 0.0)
            add_to_dangerous = max(mass_risky - current, 0.0)
            lines.append(f"  • Remove ≈ {remove_to_safe:.2f} {unit_label} for SAFE classification.")
            lines.append(f"  • Add ≈ {add_to_dangerous:.2f} {unit_label} for DANGEROUS classification.")
        else:
            remove_to_risky = max(current - mass_risky, 0.0)
            remove_to_safe = max(current - mass_safe, 0.0)
            lines.append(f"  • Remove ≈ {remove_to_risky:.2f} {unit_label} to drop to RISKY.")
            lines.append(f"  • Remove ≈ {remove_to_safe:.2f} {unit_label} to drop to SAFE.")

    lines.append("======================================")
    report = "\n".join(lines)

    return {
        "species": species,
        "production": production,
        "preparation": preparation,
        "amount_value": amount_value,
        "amount_unit": amount_unit,
        "mass_g": mass_g,
        "pred_max_metal_mg_kg": pred_max_metal,
        "metal_concentrations_mg_kg": metal_preds,
        "metal_intake_ug": metal_intake_ug,
        "metal_absorbed_ug": metal_absorbed_ug,
        "max_ratio_to_tolerance": max_ratio,
        "safety_class": safety,
        "report": report,
    }

species_options = sorted(df_target["species"].dropna().unique().tolist())

species_dd = Dropdown(
    options=species_options,
    description="Species:",
    value=species_options[0] if species_options else None,
    style={"description_width": "initial"},
    layout=Layout(width="250px"),
)

production_dd = Dropdown(
    options=["wild"],
    value="wild",
    description="Raised:",
    style={"description_width": "initial"},
    layout=Layout(width="150px"),
)

prep_dd = Dropdown(
    options=["raw", "cooked"],
    value="raw",
    description="Prep:",
    style={"description_width": "initial"},
    layout=Layout(width="140px"),
)

amount_ft = FloatText(
    value=0.5,
    description="Amount:",
    style={"description_width": "initial"},
    layout=Layout(width="200px"),
)

unit_dd = Dropdown(
    options=["lb", "kg", "g"],
    value="lb",
    description="Unit:",
    style={"description_width": "initial"},
    layout=Layout(width="120px"),
)

add_btn = Button(
    description="+",
    tooltip="Add this fish to comparison",
    layout=Layout(width="40px"),
)
clear_btn = Button(
    description="Clear",
    tooltip="Clear all comparisons",
    layout=Layout(width="70px"),
)

controls_box = HBox([species_dd, production_dd, prep_dd, amount_ft, unit_dd, add_btn, clear_btn])

out_detail = Output()
out_table = Output()
selected_results = []

species_production_options = {
    "atlantic salmon": ["farmed"],
    "coho salmon": ["wild"],
    "sockeye salmon": ["wild"],
    "salmon (generic)": ["wild", "farmed"],
    "tilapia": ["farmed"],
    "catfish": ["farmed"],
    "trout": ["wild", "farmed"],
    "pollock": ["wild"],
    "cod": ["wild"],
    "haddock": ["wild"],
    "whiting": ["wild"],
    "shrimp": ["wild", "farmed"],
    "sea bass": ["wild", "farmed"],
    "snapper": ["wild", "farmed"],
    "carp": ["wild", "farmed"],
}

def update_production_options(species):
    if species is None:
        production_dd.options = ["wild"]
        production_dd.value = "wild"
        production_dd.disabled = True
        return

    key = species.lower().strip()
    opts = species_production_options.get(key, [production_map.get(key, "wild")])

    if not isinstance(opts, (list, tuple)):
        opts = [opts]

    production_dd.options = opts
    production_dd.value = opts[0]
    production_dd.disabled = (len(opts) == 1)

def on_species_change(change):
    if change["name"] == "value" and change["new"] is not None:
        update_production_options(change["new"])

species_dd.observe(on_species_change, names="value")
update_production_options(species_dd.value)

def refresh_table():
    with out_table:
        clear_output(wait=True)
        if not selected_results:
            print("No fish added yet. Choose a species, raised type, prep, set an amount, then click '+'.")
            return

        rows = []
        for r in selected_results:
            absorbed = r["metal_absorbed_ug"]
            rows.append({
                "Species": r["species"],
                "Raised": r["production"],
                "Prep": r["preparation"],
                "Amount": f'{r["amount_value"]:.2f} {r["amount_unit"]}',
                "Hg_abs_ug": absorbed.get("mercury", np.nan),
                "Pb_abs_ug": absorbed.get("lead", np.nan),
                "As_abs_ug": absorbed.get("arsenic", np.nan),
                "Cd_abs_ug": absorbed.get("cadmium", np.nan),
                "Max_metal_mg_kg": r["pred_max_metal_mg_kg"],
                "Safety": r["safety_class"],
            })
        comp_df = pd.DataFrame(rows)
        print("=== Comparison Table (absorbed µg per week) ===")
        display(comp_df)

def on_add_clicked(b):
    if species_dd.value is None:
        return
    res = evaluate_fish_intake(
        species=species_dd.value,
        production=production_dd.value,
        preparation=prep_dd.value,
        amount_value=amount_ft.value,
        amount_unit=unit_dd.value,
        body_weight_kg=80.0,
    )
    selected_results.append(res)
    refresh_table()
    with out_detail:
        clear_output(wait=True)
        print(res["report"])

def on_clear_clicked(b):
    selected_results.clear()
    refresh_table()
    with out_detail:
        clear_output(wait=True)
        print("No fish selected yet.")

add_btn.on_click(on_add_clicked)
clear_btn.on_click(on_clear_clicked)

print("\nInteractive AI decision tool with multi-fish comparison (raw vs cooked, lb/kg/g, absorbed dose)")
display(controls_box, out_detail, out_table)
refresh_table()
