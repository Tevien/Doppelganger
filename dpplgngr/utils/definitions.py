######## MEDICATION DEFINITIONS ########

# ACE inhibitors - Plain and combinations

ace_atc = [f"C09AA{n:02d}" for n in range(1,17)] # Plain ACE inhibitors

ace_atc.extend([f"C09BA{n:02d}" for n in range(1,16)]) # ACE + diuretics

ace_atc.extend([f"C09BB{n:02d}" for n in range(2,13)]) # ACE + CCB

ace_atc.extend([f"C09BX{n:02d}" for n in range(1,6)]) # ACE + other

# Angiotensin Receptor Blockers (ARB) - Plain and combinations

arb_atc = [f"C09CA{n:02d}" for n in range(1,11)] # Plain ARB

arb_atc.extend([f"C09DA{n:02d}" for n in range(1,11)]) # ARB + diuretics

arb_atc.extend([f"C09DB{n:02d}" for n in range(1,10)]) # ARB + CCB

# ARB + other (excluding ARNI which is C09DX04)

arb_other = [f"C09DX{n:02d}" for n in range(1,8)]

arb_other.remove("C09DX04") # Remove ARNI

arb_atc.extend(arb_other)

# Renin-Angiotensin System (RAS) inhibitors - broader category

ras_atc = ["C09AA", "C09B", "C09CA", "C09D", "C09X"] # Parent codes

# Angiotensin Receptor-Neprilysin Inhibitor (ARNI)

arni_atc = ["C09DX04"] # sacubitril/valsartan

# Beta-blockers - comprehensive list

beta_atc = {

"Atenolol": "C07AB03",

"Bisoprolol": "C07AB07",

"Metoprolol": "C07AB02",

"Propranolol": "C07AA05",

"Carvedilol": "C07AG02",

"Nebivolol": "C07AB12"

}

# Beta-blockers - all categories

beta_all_atc = ["C07AA", "C07AB", "C07AG"] # Plain beta blockers

beta_all_atc.extend(["C07BA", "C07BB", "C07BG"]) # + thiazides

beta_all_atc.extend(["C07CA", "C07CB", "C07CG"]) # + diuretics

beta_all_atc.extend(["C07DA", "C07DB", "C07DG"]) # + other

beta_all_atc.extend(["C07FB", "C07FX"]) # + RAS

# Mineralocorticoid Receptor Antagonists (MRA)

mra_atc = [f"C03DA{n:02d}" for n in range(1,6)]

mra_atc.extend([f"C03EA{n:02d}" for n in range(1,15)])

mra_atc.extend(["C03EB01", "C03EB02"]) # combination potassium sparing

# Diuretics - comprehensive

diuretics_atc = ["C03A", "C03C", "C03D"] # Main categories

# Loop diuretics

loop_diuretics = [f"C03CA{n:02d}" for n in range(1,5)]

loop_diuretics.extend(["C03CB01", "C03CB02", "C03CC01", "C03CC02", "C03CX01"])

# Thiazide diuretics

thiazide_diuretics = ["C03A"]

# Potassium-sparing agents

potassium_sparing = ["C03D"]

# SGLT-2 inhibitors

sglt2_hf_atc = ["A10BX09", "A10BK01", "A10BX12", "A10BK03"] # For HF patients

sglt2_other_atc = ["A10BX11", "A10BK02", "A10BK04", "A10BK05", "A10BK06"] # Other

# Anticoagulants

anticoagulant_atc = ["B01AA", "B01AB", "B01AX05", "B01AX04", "B01AF", "B01AE",

"B01AX01", "B01AD12", "B01AB02"]

# Antiplatelet agents

antiplatelet_atc = ["B01AC13", "B01AC17", "B01AC16", "B01AC04", "B01AC22", "B01AC05",

"B01AC25", "B01AC24", "B01AC19", "B01AC11", "B01AC21", "B01AC06",

"B01AC15", "B01AC08", "B01AC10", "B01AC18", "B01AC07", "B01AC03",

"B01AC23", "B01AC01", "B01AC02", "B01AC26"]

# Statins

statins_atc = ["C10AA", "C10BA", "C10BX"]

# Specific statins

specific_statins = {

"Atorvastatin": "C10AA05",

"Simvastatin": "C10AA01",

"Rosuvastatin": "C10AA07",

"Pravastatin": "C10AA03"

}

# Digitalis glycosides

digitalis_atc = [f"C01AA{n:02d}" for n in range(1,9)]

# Ivabradine

ivabradine_atc = ["C01EB17"]

# Calcium Channel Blockers

ccb_atc = ["C08C", "C08D", "C08E", "C08GA"]

######## DISEASE DEFINITIONS ########

# Heart Failure - ICD-10 (EXPANDED with complete subcodes)
heart_failure_icd10 = [
    # I50 - Heart failure (complete subcodes)
    "I50.1", "I50.20", "I50.21", "I50.22", "I50.23",  # Left ventricular failure, systolic HF
    "I50.30", "I50.31", "I50.32", "I50.33",  # Diastolic HF
    "I50.40", "I50.41", "I50.42", "I50.43",  # Combined systolic and diastolic HF
    "I50.9",  # Heart failure, unspecified
    # Other related codes
    "I11.0", "I13.0", "I13.2", "I26.0", "I25.5", "I09.81", "I97.13"
]

# Heart Failure - ICD-9 (EXPANDED with complete subcodes)
heart_failure_icd9 = [
    # 428 - Heart failure (complete subcodes)
    "428.0", "428.1", "428.20", "428.21", "428.22", "428.23",
    "428.30", "428.31", "428.32", "428.33", "428.40", "428.41", 
    "428.42", "428.43", "428.9",
    # Other related codes
    "404.01", "404.03", "404.11", "404.13", "404.91", "404.93", 
    "402.01", "402.11", "402.91", "415.0", "418.8", "398.91", "429.4"
]

# Atrial Fibrillation/Flutter - ICD-10 (EXPANDED with complete subcodes)
af_icd_10 = [
    "I48.0",   # Paroxysmal atrial fibrillation
    "I48.11",  # Longstanding persistent atrial fibrillation
    "I48.19",  # Other persistent atrial fibrillation
    "I48.20",  # Chronic atrial fibrillation, unspecified
    "I48.21",  # Permanent atrial fibrillation
    "I48.3",   # Typical atrial flutter
    "I48.4",   # Atypical atrial flutter
    "I48.91",  # Unspecified atrial fibrillation
    "I48.92"   # Atrial flutter, unspecified
]

# Atrial Fibrillation/Flutter - ICD-9 (complete)
af_icd_9 = ["427.3", "427.31", "427.32"]

# Angina Pectoris - ICD-10 (EXPANDED with complete subcodes)
angina_icd10 = [
    "I20.0",  # Unstable angina
    "I20.1",  # Angina pectoris with documented spasm
    "I20.2",  # Refractory angina pectoris
    "I20.8",  # Other forms of angina pectoris
    "I20.9"   # Angina pectoris, unspecified
]

# Angina Pectoris - ICD-9 (EXPANDED with complete subcodes)
angina_icd9 = [
    # 411 - Other acute and subacute forms of ischemic heart disease
    "411.0", "411.1", "411.81", "411.89",
    # 413 - Angina pectoris  
    "413.0", "413.1", "413.9"
]

# Myocardial Infarction - ICD-10 (EXPANDED with complete subcodes)
mi_icd10 = [
    # I21 - Acute myocardial infarction
    "I21.1", "I21.2", "I21.9",  # STEMI of anterior wall
    "I21.11", "I21.19",            # STEMI of inferior wall
    "I21.21", "I21.29",            # STEMI of other sites
    "I21.3",                       # STEMI of unspecified site
    "I21.4",                       # Non-ST elevation (NSTEMI) myocardial infarction
    "I21.9", "I21.A1",             # Acute MI unspecified, Type 2 MI
    # I22 - Subsequent myocardial infarction
    "I22.0", "I22.1", "I22.2", "I22.8", "I22.9"
]

# Myocardial Infarction - ICD-9 (EXPANDED with complete subcodes)
mi_icd9 = [
    # 410 - Acute myocardial infarction (all subcodes)
    "410.00", "410.01", "410.02",  # Anterolateral wall
    "410.10", "410.11", "410.12",  # Other anterior wall
    "410.20", "410.21", "410.22",  # Inferolateral wall
    "410.30", "410.31", "410.32",  # Inferoposterior wall
    "410.40", "410.41", "410.42",  # Other inferior wall
    "410.50", "410.51", "410.52",  # Other lateral wall
    "410.60", "410.61", "410.62",  # True posterior wall
    "410.70", "410.71", "410.72",  # Subendocardial infarction
    "410.80", "410.81", "410.82",  # Other specified sites
    "410.90", "410.91", "410.92",  # Unspecified site
    # 412 - Old myocardial infarction
    "412"
]

# Chronic Ischemic Heart Disease
chron_ischemic_hd_icd_10 = ["I20.0", "I20.1", "I20.2", "I20.8", "I20.9",
                            "I21.1", "I21.2", "I21.9", "I21.11", "I21.19", "I21.21", "I21.29", "I21.3", "I21.4", "I21.9", "I21.A1",
                            "I22.0", "I22.1", "I22.2", "I22.8", "I22.9",
                            "I23.0", "I23.1", "I23.2", "I23.3", "I23.4", "I23.5", "I23.6", "I23.7", "I23.8",
                            "I24.0", "I24.1", "I24.8", "I24.9",
                            "I25.10", "I25.110", "I25.111", "I25.118", "I25.119", "I25.2", "I25.3", "I25.5", "I25.6", "I25.8", "I25.9"]

chron_ischemic_hd_icd_9 = ["410.00", "410.01", "410.02", "410.10", "410.11", "410.12", "410.20", "410.21", "410.22",
                          "410.30", "410.31", "410.32", "410.40", "410.41", "410.42", "410.50", "410.51", "410.52",
                          "410.60", "410.61", "410.62", "410.70", "410.71", "410.72", "410.80", "410.81", "410.82",
                          "410.90", "410.91", "410.92", "411.0", "411.1", "411.81", "411.89", "412", 
                          "413.0", "413.1", "413.9", "414.0", "414.2", "414.3", "414.4", "414.8", "414.9", "429.2"]

# Hypertension - ICD-10 (EXPANDED with complete subcodes)
hypertension_icd10 = [
    "I10",     # Essential hypertension
    # I11 - Hypertensive heart disease
    "I11.0", "I11.9",
    # I12 - Hypertensive chronic kidney disease
    "I12.0", "I12.9",
    # I13 - Hypertensive heart and chronic kidney disease
    "I13.0", "I13.10", "I13.11", "I13.2",
    # I15 - Secondary hypertension
    "I15.0", "I15.1", "I15.2", "I15.8", "I15.9"
]

# Hypertension - ICD-9 (EXPANDED with complete subcodes)
hypertension_icd9 = [
    # 401 - Essential hypertension
    "401.0", "401.1", "401.9",
    # 402 - Hypertensive heart disease
    "402.00", "402.01", "402.10", "402.11", "402.90", "402.91",
    # 403 - Hypertensive chronic kidney disease
    "403.00", "403.01", "403.10", "403.11", "403.90", "403.91",
    # 404 - Hypertensive heart and chronic kidney disease
    "404.00", "404.01", "404.02", "404.03", "404.10", "404.11", "404.12", "404.13",
    "404.90", "404.91", "404.92", "404.93",
    # 405 - Secondary hypertension
    "405.01", "405.09", "405.11", "405.19", "405.91", "405.99"
]

# Diabetes Mellitus - ICD-10 (EXPANDED with complete subcodes for main categories)
diabetes_icd10 = [
    # E10 - Type 1 diabetes mellitus (key subcodes)
    "E10.9", "E10.10", "E10.21", "E10.22", "E10.29", "E10.40", "E10.42", "E10.43", "E10.49",
    "E10.51", "E10.59", "E10.621", "E10.622", "E10.628", "E10.630", "E10.638", "E10.641",
    "E10.649", "E10.65", "E10.69", "E10.8",
    # E11 - Type 2 diabetes mellitus (key subcodes)
    "E11.9", "E11.0", "E11.1", "E11.10", "E11.11", "E11.21", "E11.22", "E11.29", 
    "E11.311", "E11.319", "E11.321", "E11.329", "E11.331", "E11.339", "E11.351", "E11.359",
    "E11.40", "E11.41", "E11.42", "E11.43", "E11.44", "E11.49", "E11.51", "E11.52", "E11.59",
    "E11.610", "E11.618", "E11.620", "E11.621", "E11.622", "E11.628", "E11.630", "E11.638",
    "E11.641", "E11.649", "E11.65", "E11.69", "E11.8",
    # E12 - Malnutrition-related diabetes mellitus (key subcodes)
    "E12.9", "E12.0", "E12.1", "E12.10", "E12.11", "E12.21", "E12.22", "E12.29",
    "E12.40", "E12.41", "E12.42", "E12.43", "E12.44", "E12.49", "E12.51", "E12.52", "E12.59",
    "E12.621", "E12.622", "E12.628", "E12.641", "E12.649", "E12.65", "E12.69", "E12.8",
    # E13 - Other specified diabetes mellitus (key subcodes)
    "E13.9", "E13.0", "E13.1", "E13.10", "E13.11", "E13.21", "E13.22", "E13.29",
    "E13.40", "E13.41", "E13.42", "E13.43", "E13.44", "E13.49", "E13.51", "E13.52", "E13.59",
    "E13.621", "E13.622", "E13.628", "E13.641", "E13.649", "E13.65", "E13.69", "E13.8",
    # E14 - Unspecified diabetes mellitus (key subcodes)
    "E14.9", "E14.0", "E14.1", "E14.10", "E14.11", "E14.21", "E14.22", "E14.29",
    "E14.40", "E14.41", "E14.42", "E14.43", "E14.44", "E14.49", "E14.51", "E14.52", "E14.59",
    "E14.621", "E14.622", "E14.628", "E14.641", "E14.649", "E14.65", "E14.69", "E14.8"
]

# Diabetes Mellitus - ICD-9 (EXPANDED with complete subcodes)
diabetes_icd9 = [
    # 249 - Secondary diabetes mellitus (all subcodes)
    "249.00", "249.01", "249.10", "249.11", "249.20", "249.21", "249.30", "249.31",
    "249.40", "249.41", "249.50", "249.51", "249.60", "249.61", "249.70", "249.71",
    "249.80", "249.81", "249.90", "249.91",
    # 250 - Diabetes mellitus (all subcodes)
    "250.00", "250.01", "250.02", "250.03", "250.10", "250.11", "250.12", "250.13",
    "250.20", "250.21", "250.22", "250.23", "250.30", "250.31", "250.32", "250.33",
    "250.40", "250.41", "250.42", "250.43", "250.50", "250.51", "250.52", "250.53",
    "250.60", "250.61", "250.62", "250.63", "250.70", "250.71", "250.72", "250.73",
    "250.80", "250.81", "250.82", "250.83", "250.90", "250.91", "250.92", "250.93"
]

# Type 2 Diabetes - detailed (keeping existing detailed list)
t2dm_codes = [
"E11.9", # without complications
"E11.21", # diabetic nephropathy
"E11.22", # diabetic CKD
"E11.29", # other kidney complication
"E11.31", "E11.32", "E11.33", "E11.34", "E11.35", "E11.36", "E11.39", # retinopathy
"E11.40", "E11.41", "E11.42", "E11.43", "E11.44", "E11.49", # neuropathy
"E11.51", "E11.52", "E11.59", # circulatory complications
"E11.65", # hyperglycemia
"E11.69", # other specified complication
"E11.8" # unspecified complications
]

# Chronic Kidney Disease - ICD-10 (EXPANDED with complete subcodes)
ckd_icd10 = [
    # N18 - Chronic kidney disease (complete subcodes)
    "N18.1",  # CKD, stage 1
    "N18.2",  # CKD, stage 2 (mild)
    "N18.3",  # CKD, stage 3 (moderate)
    "N18.30", "N18.31", "N18.32",  # CKD stage 3 subdivisions
    "N18.4",  # CKD, stage 4 (severe)
    "N18.5",  # CKD, stage 5
    "N18.6",  # End stage renal disease
    "N18.9",  # CKD, unspecified
    # N19 - Unspecified kidney failure
    "N19"
]

# Chronic Kidney Disease - ICD-9 (EXPANDED with complete subcodes)
ckd_icd9 = [
    # 585 - Chronic kidney disease (complete subcodes)
    "585.1", "585.2", "585.3", "585.4", "585.5", "585.6", "585.9",
    # 586 - Renal failure, unspecified
    "586"
]

# Stroke
stroke_icd10 = ["I61", "I62", "I63", "I64", "I60"]

stroke_icd9 = ["430", "431", "432", "433.01", "433.11", "433.21", "433.31",
"433.81", "433.91", "434.01", "434.11", "434.91", "436"]

# Transient Ischemic Attack
tia_icd10 = ["G45.8", "G45.9", "I63.9"]

tia_icd9 = ["435", "435.8"]

# Peripheral Artery Disease
pad_icd10 = ["I73", "I70", "I71", "I72", "I74", "I75", "I77", "I78", "I79"]

pad_icd9 = ["443", "440", "441", "442", "444", "445", "446", "447", "448"]

# Cardiomyopathy
cardiomyopathy_icd10 = ["I42", "I43"]

cardiomyopathy_icd9 = ["425"]

# Dilated Cardiomyopathy
dilated_cm_icd10 = ["I42.0"]

dilated_cm_icd9 = ["425.4"]

# Valvular Disease
valvular_icd10 = ["A520", "I05", "I06", "I07", "I08", "I09.1", "I09.8", "I34", "I35",
"I36", "I37", "I38", "I39", "Q23.0", "Q23.1", "Q23.2", "Q23.3",
"Z95.2", "Z95.3", "Z95.4"]

valvular_icd9 = ["093.2", "746.3", "746.4", "746.5", "746.6", "V42.2", "V43.3",
"394", "395", "396", "397", "424", "746"]

# COPD
copd_icd10 = ["J40", "J41", "J42", "J43", "J44"]

copd_icd9 = ["490", "491", "492", "494", "495", "496"]

# Depression
depression_icd10 = ["F20.4", "F31.3", "F31.4", "F31.5", "F32", "F33", "F34.1",
"F41.2", "F43.2"]

depression_icd9 = ["296.2", "296.3", "296.5", "300.4", "309", "311"]

# Dementia
dementia_icd10 = ["F00", "F01", "F02", "F03", "F05.1", "G30", "G31.1"]

dementia_icd9 = ["290"]

# Liver Disease
liver_disease_icd10 = ["B18", "I85", "I86.4", "I98.2", "K70", "K71.1", "K71.3",
"K71.4", "K71.5", "K71.7", "K72", "K73", "K74", "K76.0",
"K76.2", "K76.3", "K76.4", "K76.5", "K76.6", "K76.7",
"K76.8", "K76.9", "Z94.4"]

liver_disease_icd9 = ["070.22", "070.23", "070.32", "070.33", "070.44", "070.54",
"070.6", "070.9", "456.0", "456.1", "456.2", "570", "571",
"572.2", "572.3", "572.4", "572.5", "572.6", "572.7", "572.8",
"573.3", "573.4", "573.8", "573.9", "V42.7"]

######## OUTCOME DEFINITIONS ########

# Heart Failure Hospitalization/ED Visit
hf_hosp_icd10 = ["I50.1", "I50.20", "I50.21", "I50.22", "I50.23", "I50.30", "I50.31", 
                 "I50.32", "I50.33", "I50.40", "I50.41", "I50.42", "I50.43", "I50.9",
                 "I11.0", "I13.0", "I13.2", "I26.0", "I09.81", "I97.13"]

hf_hosp_icd9 = ["428.0", "428.1", "428.20", "428.21", "428.22", "428.23", "428.30", 
                "428.31", "428.32", "428.33", "428.40", "428.41", "428.42", "428.43", "428.9",
                "404.01", "404.03", "404.11", "404.13", "404.91", "404.93",
                "402.01", "402.11", "402.91", "415.0"]

# Major Adverse Cardiovascular Events (MACE) components
mace_components = {
"acute_mi": mi_icd10,
"stroke": stroke_icd10,
"hf_hosp": hf_hosp_icd10
}

######## MAPPING DICTIONARY ########

defs_map = {
# Medications
"ace": ace_atc,
"arb": arb_atc,
"ras": ras_atc,
"arni": arni_atc,
"beta": list(beta_atc.values()),
"beta_all": beta_all_atc,
"mra": mra_atc,
"diuretics": diuretics_atc,
"loop_diuretics": loop_diuretics,
"thiazide_diuretics": thiazide_diuretics,
"sglt2_hf": sglt2_hf_atc,
"sglt2_other": sglt2_other_atc,
"anticoagulant": anticoagulant_atc,
"antiplatelet": antiplatelet_atc,
"statins": statins_atc,
"digitalis": digitalis_atc,
"ivabradine": ivabradine_atc,
"ccb": ccb_atc,

# Cardiovascular Conditions
"heart_failure": heart_failure_icd10,
"af": af_icd_10,
"angina": angina_icd10,
"acute_mi": mi_icd10,
"chronic_ischemic_hd": chron_ischemic_hd_icd_10,
"hypertension": hypertension_icd10,
"stroke": stroke_icd10,
"tia": tia_icd10,
"pad": pad_icd10,
"cardiomyopathy": cardiomyopathy_icd10,
"dilated_cm": dilated_cm_icd10,
"valvular": valvular_icd10,

# Other Conditions
"diabetes": diabetes_icd10,
"t2dm": t2dm_codes,
"ckd": ckd_icd10,
"copd": copd_icd10,
"depression": depression_icd10,
"dementia": dementia_icd10,
"liver_disease": liver_disease_icd10,

# Outcomes
"hf_hosp": hf_hosp_icd10,
"mace": mace_components
}

def make_classification_map(l_keys):
    """
    Create a classification map from a list of keys.
    The keys are used to create a dictionary where the key is the classification name
    and the value is a list of values that belong to that classification.
    """
    classification_map = {}
    for key in l_keys:
        if key in defs_map:
            classification_map[key] = defs_map[key]
        else:
            raise ValueError(f"Key {key} not found in defs_map")
    return classification_map

def get_hf_medication_bundle():
    """
    Return the guideline-directed medical therapy (GDMT) bundle for heart failure.
    Based on current ESC guidelines.
    """
    return {
        "ace_arb_arni": ace_atc + arb_atc + arni_atc,
        "beta_blockers": beta_all_atc,
        "mra": mra_atc,
        "sglt2i": sglt2_hf_atc
        }

def get_ckd_stages():
    """
    Return eGFR thresholds for CKD staging as per protocol.
    """
    return {
    "normal": ">=90",
    "mild": "60-89",
    "moderate": "15-59",
    "advanced": "<15"
    }

def get_hyperkalemia_stages():
    """
    Return potassium level thresholds for hyperkalemia staging as per protocol.
    """
    return {
    "mild": "5.0-5.5 mmol/L",
    "moderate": "5.5-6.0 mmol/L",
    "severe": ">6.0 mmol/L"
    }
