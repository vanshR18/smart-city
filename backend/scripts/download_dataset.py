"""
download_dataset.py
───────────────────
Builds the NLP training dataset in two steps:

1. Downloads the "NLP Getting Started" disaster tweets dataset from HuggingFace
   (this is the standard emergency/non-emergency tweet dataset — 7,613 tweets,
   already labeled real_disaster vs not).

2. Augments it with our own Lucknow-specific multi-class sentences so the
   model learns ACCIDENT / FIRE / FLOOD / CRIME / CROWD / MEDICAL / NORMAL
   instead of just binary disaster / no-disaster.

Why augment?
The Kaggle dataset is binary. We need 7 classes. Augmentation bridges this.
The model learns general emergency language from Kaggle + local Indian
context from our synthetic sentences.

Output:
  app/nlp/data/train.csv
  app/nlp/data/val.csv
  app/nlp/data/test.csv

Run:
  cd backend
  python scripts/download_dataset.py
"""

import os
import sys
import random
import pandas as pd
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table

# ── make sure app/ is importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()
DATA_DIR = Path(__file__).parent.parent / "app" / "nlp" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Label map ─────────────────────────────────────────────────────────────────
LABEL2ID = {
    "ACCIDENT": 0,
    "FIRE":     1,
    "FLOOD":    2,
    "CRIME":    3,
    "CROWD":    4,
    "MEDICAL":  5,
    "NORMAL":   6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# ── Lucknow-specific synthetic sentences ─────────────────────────────────────
# ~60 sentences per class. Mix of Hindi-English (Hinglish), pure English,
# and short telegraphic styles (like real tweets and WhatsApp messages).

SYNTHETIC_DATA = {
    "ACCIDENT": [
        "Bada accident hua hai Hazratganj crossing pe, ambulance bulao jaldi",
        "Major road accident near Charbagh station, multiple vehicles involved",
        "Truck overturned on Kanpur road blocking all traffic",
        "Bike collision with auto near Alambagh chowk, person injured badly",
        "Hit and run case reported near Gomti Nagar flyover",
        "Two cars collided head on at Transport Nagar crossing",
        "Pedestrian hit by speeding bus near Kaiserbagh",
        "Road accident on ring road, one person critical, need help",
        "Terrible crash at Thakurganj signal, people trapped inside car",
        "Vehicle overturned in rain near Chinhat, driver unconscious",
        "Auto rickshaw accident near Mahanagar metro station",
        "Multiple injured in pile up on Lucknow Kanpur highway",
        "School bus accident reported near Indira Nagar, children hurt",
        "Drunk driver caused accident near Gomti Nagar extension",
        "Serious accident at Faizabad road crossing, police needed",
        "Car fell into ditch near Amausi airport road at night",
        "Motorcycle accident near Raja Ji Puram, rider bleeding",
        "Collision between truck and bus near Charbagh underpass",
        "Road caved in causing accident in Aliganj colony",
        "Three vehicles piled up at Hazratganj main crossing today",
        "ACCIDENT: car vs pedestrian near Vikas Nagar market, urgent",
        "Train almost derailed near Charbagh station due to flooding",
        "Bike skidded in rain near Gomti Nagar, rider badly hurt",
        "Accident on elevated road causing 2km traffic jam",
        "Man fell from overbridge near Transport Nagar seriously injured",
        "Oil tanker overturned near Alambagh causing major road block",
        "Child hit by car while crossing road in Indira Nagar",
        "Accident at Hazratganj T-point, multiple people hurt",
        "Car vs e-rickshaw accident near Thakurganj market area",
        "Police needed at Chinhat road accident scene immediately",
    ],
    "FIRE": [
        "Building mein aag lag gayi Kaiserbagh mein, fire brigade bulao",
        "Fire broke out in market near Hazratganj, spreading to nearby shops",
        "Short circuit fire in residential colony Indira Nagar, evacuate now",
        "Massive fire at garment factory Transport Nagar, thick smoke visible",
        "Gas cylinder blast in Thakurganj area, multiple families affected",
        "Fire in slum area near Charbagh station, many huts burning",
        "Electrical fire in commercial building Gomti Nagar, building evacuated",
        "FIRE ALERT: old paper godown burning near Kaiserbagh, help needed",
        "Chemical factory fire in Amausi, toxic smoke spreading",
        "Fire on top floor of hotel near Hazratganj, guests trapped",
        "Kitchen fire spread to entire apartment in Mahanagar",
        "Petrol pump fire reported near Alambagh, dangerous situation",
        "Fire in jhuggis near railway track Charbagh, 50 families affected",
        "Transformer blast caused fire in Indira Nagar residential area",
        "Wedding hall caught fire in Gomti Nagar, people evacuating",
        "Fire in warehouse near Transport Nagar, firefighters on way",
        "Short circuit in hospital ward causing fire, patients shifted",
        "Smoke seen from building in Aliganj, may be fire inside",
        "Gas leak ignited causing explosion and fire in Thakurganj",
        "Market arson suspected near Kaiserbagh, police and fire brigade called",
        "Fire broke out in government office building Hazratganj",
        "Electrical substation fire causing power outage in Gomti Nagar",
        "Massive blaze at cold storage near Transport Nagar, contained now",
        "Fire in school building Indira Nagar at night, no injuries",
        "Temple caught fire in old Lucknow, historical structure at risk",
        "LPG cylinder explosion caused fire in Alambagh home",
        "Wildfire like situation in dry grass area near Chinhat",
        "Vehicle caught fire on ring road near Mahanagar",
        "Fire at printing press near Charbagh, workers escaped safely",
        "Smoke alert in shopping mall Gomti Nagar, evacuation in progress",
    ],
    "FLOOD": [
        "Gomti river water level rising dangerously, low areas alert",
        "Heavy waterlogging in Charbagh underpass, vehicles completely submerged",
        "Flood situation in Chinhat area, water entering homes of residents",
        "Alambagh underpass flooded after heavy rain, avoid this route",
        "Rescue teams deployed in Amausi after flooding situation worsened",
        "Roads completely flooded in Indira Nagar colony after rain",
        "River embankment breach reported near Gomti Nagar extension",
        "FLOOD WARNING: Gomti river above danger mark, evacuate riverside areas",
        "Waterlogging so bad in Hazratganj that vehicles are floating",
        "People stranded on rooftops in flood affected areas of Lucknow",
        "Flood relief camps opened in Thakurganj and Kaiserbagh",
        "Bridge submerged, connectivity cut off in Chinhat area",
        "NDRF teams called for flood rescue operation near Gomti river",
        "Low lying areas of Alambagh facing severe waterlogging problem",
        "Rain water entered hospital ground floor in Mahanagar area",
        "Colony roads turned into rivers after 3 hours of heavy rain",
        "Drain overflow causing flooding in Indira Nagar sector 3",
        "Multiple localities cut off due to flood near Amausi",
        "Flood water receding in Gomti Nagar but roads still blocked",
        "Animals stranded in flood water near Chinhat village",
        "Submerged road near Charbagh causing 5km traffic jam",
        "Flash flood warning issued for low lying areas of Lucknow",
        "School closed due to flooding in Thakurganj area",
        "Pump houses failed causing severe waterlogging in Kaiserbagh",
        "Flood water reached first floor of apartments in Aliganj",
        "Gomti nadi mein bahut zyada paani, khatre ki ghanti baj rahi",
        "Puri colony mein paani bhar gaya, log ghar chhod rahe hain",
        "Nali overflow se sadak pe 3 feet paani, koi madad karo",
        "Flood rescue boat deployed in Mahanagar residential area",
        "Relief material distribution started at Alambagh flood camp",
    ],
    "CRIME": [
        "Chain snatching reported near Hazratganj metro station area",
        "Armed robbery at jewellery shop Kaiserbagh, police needed urgently",
        "Suspicious person with weapon seen near Mahanagar park at night",
        "Mobile snatching incident near Charbagh bus stand, thief escaped",
        "Eve teasing complaint near Gomti Nagar mall, youth detained",
        "House robbery in Indira Nagar, family was not home",
        "Kidnapping attempt foiled near school in Aliganj, parents alert",
        "Drug trafficking caught by police near Transport Nagar",
        "ATM robbery attempted in Thakurganj at midnight",
        "Car theft reported in Vikas Nagar parking area last night",
        "Group of men creating trouble near Kaiserbagh chowk",
        "Stabbing incident near Charbagh station, victim hospitalised",
        "Fraud case reported at Hazratganj bank branch",
        "Molestation case in Gomti Nagar park reported by victim",
        "Gang fight reported near Thakurganj, police called",
        "Pickpocket gang active at Charbagh railway station platform",
        "Online fraud victim reports loss of 5 lakhs in Indira Nagar",
        "Violent altercation between neighbours in Mahanagar colony",
        "Theft from temple in old Lucknow, CCTV footage sought",
        "Domestic violence complaint from Alambagh residential area",
        "Chain se loot hua mere dost ko Hazratganj ke paas",
        "Chori ho gayi dukaan mein raat ko, police ko phone karo",
        "Suspicious bag found near Kaiserbagh, bomb squad called",
        "Road rage incident turned violent near Gomti Nagar",
        "Extortion threat given to businessman in Transport Nagar",
        "Hit and run driver absconded after accident near Chinhat",
        "Fake police officer arrested in Mahanagar area",
        "Mobile shop looted at gunpoint in Thakurganj market",
        "Women harassed near Aliganj metro station at evening",
        "Cybercrime complaint filed by student duped of admission fee",
    ],
    "CROWD": [
        "Massive crowd gathering at Hazratganj for election rally today",
        "Stampede like situation at Charbagh station, train delayed",
        "Huge crowd at IG Pratishtha Marg, traffic completely halted",
        "Religious procession creating crowd situation in old city area",
        "Flash mob of protestors blocking road near Kaiserbagh",
        "Unmanageable crowd at Gomti Nagar shopping mall on sale day",
        "Crowd surging at political event near Vidhaan Sabha, unsafe",
        "Mela crowd causing chaos near Hazratganj, police deployed",
        "College students protest blocking road in Indira Nagar",
        "Wedding procession blocking major road near Thakurganj",
        "Crowd gathered after accident at Alambagh, obstructing rescue",
        "Festival crowd at Hazratganj too dense, crush risk high",
        "Lakh plus crowd at Gomti riverfront for event, manage now",
        "Public agitation near Collectorate creating law order problem",
        "Long queue fight at ration shop in Mahanagar turning violent",
        "Youth crowd rally at Eco Garden turned aggressive suddenly",
        "Dussehra mela crowd out of control near Rana Pratap Marg",
        "Rush at Charbagh station during festival season, stampede fear",
        "Unexpected large crowd at liquor shop before dry day",
        "IPL victory celebration crowd blocking Hazratganj road",
        "Bhaari bheed jam gayi Charbagh station par, halat kharab",
        "Protest march turned chaotic near Gomti Nagar police station",
        "Overcrowded bus caused passenger injury near Alambagh",
        "Funeral procession of 10000 people blocking Faizabad road",
        "Flash sale crowd fighting at mall in Gomti Nagar extension",
        "Holi celebration crowd blocking emergency vehicle in Indira Nagar",
        "Cracker market too crowded near Kaiserbagh, safety concern",
        "Cricket match crowd spillover on road near KANA stadium",
        "Kanwar yatra crowd of lakhs on roads, traffic diverted",
        "Crowd loot at PDS shop Thakurganj, police called",
    ],
    "MEDICAL": [
        "Medical emergency on Mahanagar road, person collapsed suddenly",
        "Patient critical condition near KGMU hospital, ambulance stuck in traffic",
        "Old man collapsed at Hazratganj market, need ambulance immediately",
        "Road accident victim bleeding heavily near Charbagh, no help yet",
        "Multiple people feeling unwell at food fair Gomti Nagar",
        "Heart attack reported in Indira Nagar colony, calling 108",
        "Pregnant woman in labour stuck in traffic jam Alambagh",
        "Child swallowed something, parents panicking near Mahanagar hospital",
        "Mass food poisoning at wedding function in Kaiserbagh area",
        "Dengue patient in critical condition in Thakurganj, need blood",
        "Snake bite victim being rushed to hospital from Chinhat village",
        "Gas leak victim unconscious, need medical help in Amausi",
        "Suicide attempt near Gomti riverbank, ambulance called",
        "Electric shock victim near Transport Nagar, serious burns",
        "Drowning victim pulled from Gomti river near Laxman Mela ground",
        "Dialysis patient unable to reach hospital due to road blockage",
        "Diabetic patient collapsed in Vikas Nagar market area",
        "Burn victim from kitchen accident in Aliganj needs help",
        "Multiple people injured in building collapse in old Lucknow",
        "Asthma attack patient stuck in traffic, needs immediate help",
        "Bachcha kho gaya hospital ke paas, parents dhoondh rahe hain",
        "Severe allergic reaction at restaurant Gomti Nagar, help needed",
        "Mentally disturbed person creating scene near Charbagh station",
        "Premature delivery happening at home in Thakurganj, call doctor",
        "Medical camp overwhelmed with patients in flood affected area",
        "Cholera outbreak suspected in Chinhat slum area",
        "Overdose patient found unconscious near Kaiserbagh park",
        "Accident victim not getting blood type O+ near KGMU",
        "Old woman fell down stairs in Mahanagar, may have fracture",
        "Ambulance stuck at Hazratganj signal for 15 minutes, patient critical",
    ],
    "NORMAL": [
        "Traffic moving smoothly on Hazratganj road this morning",
        "All clear in Gomti Nagar, no incidents reported today",
        "Routine patrol completed in Alambagh area, situation normal",
        "Weather clear, roads safe near Indira Nagar colony",
        "No emergency situations reported this hour in Lucknow",
        "Minor traffic slowdown near Charbagh, moving slowly but fine",
        "Police flag march in Kaiserbagh area, precautionary measure",
        "Water supply restored in Mahanagar after maintenance work",
        "Power back in Thakurganj after 2 hour outage, normal now",
        "Street lights repaired in Gomti Nagar sector 4, area bright",
        "Garbage collected from Indira Nagar colony today morning",
        "Road work completed near Alambagh, traffic flowing normally",
        "Test alert: system check by smart city control room",
        "Routine check at Charbagh railway station, everything fine",
        "Weather update: light rain expected, roads clear currently",
        "Market day in Aliganj, slight congestion but manageable",
        "School traffic managed well near Vikas Nagar this morning",
        "No waterlogging reported after rain in Gomti Nagar area",
        "Pothole repair work going on near Mahanagar, slow traffic only",
        "Public event at Eco Garden going on peacefully",
        "Bus service on time today at Charbagh station platform 4",
        "Power maintenance work in Thakurganj from 10am to 12pm",
        "Water tanker supply resumed in Amausi colony",
        "Traffic police managing festival crowd well in Hazratganj",
        "All roads open after yesterday's waterlogging, situation clear",
        "Routine police check near Kaiserbagh, nothing suspicious found",
        "Hospital OPD running normally at KGMU today",
        "Airport road clear, flight operations normal at Amausi",
        "Morning walk safe in Gomti Nagar riverside area today",
        "No complaints received from Indira Nagar last night, peaceful",
    ],
}


def build_dataframe() -> pd.DataFrame:
    """Converts SYNTHETIC_DATA dict into a DataFrame."""
    rows = []
    for label, sentences in SYNTHETIC_DATA.items():
        for text in sentences:
            rows.append({"text": text.strip(), "label": LABEL2ID[label], "label_name": label})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle
    return df


def augment_with_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to pull the HuggingFace NLP disaster tweets dataset and map
    real_disaster=1 → splits into ACCIDENT/FIRE/FLOOD/CRIME based on keywords,
    real_disaster=0 → NORMAL.

    If the download fails (no internet), we skip augmentation gracefully.
    """
    try:
        from datasets import load_dataset
        logger.info("Downloading NLP disaster tweets dataset from HuggingFace...")
        ds = load_dataset("nlp-disaster-tweets-2020", trust_remote_code=True)
        logger.success("Downloaded HuggingFace dataset")
    except Exception:
        try:
            # fallback: smaller built-in dataset
            from datasets import load_dataset
            ds = load_dataset("disaster_response_messages", split="train")
        except Exception as e:
            logger.warning(f"Could not download external dataset ({e}). Using synthetic data only.")
            return df

    try:
        hf_df = ds["train"].to_pandas() if hasattr(ds, "__getitem__") else ds.to_pandas()

        # Map to our labels using keyword heuristics on the tweet text
        def map_label(row):
            text_lower = str(row.get("text", "")).lower()
            target = row.get("target", 0)
            if target == 0:
                return LABEL2ID["NORMAL"]
            if any(w in text_lower for w in ["fire", "burn", "flame", "blaze"]):
                return LABEL2ID["FIRE"]
            if any(w in text_lower for w in ["flood", "rain", "water", "drown"]):
                return LABEL2ID["FLOOD"]
            if any(w in text_lower for w in ["crash", "accident", "collision", "wreck"]):
                return LABEL2ID["ACCIDENT"]
            if any(w in text_lower for w in ["shoot", "gun", "attack", "crime", "rob"]):
                return LABEL2ID["CRIME"]
            if any(w in text_lower for w in ["crowd", "stampede", "protest", "rally"]):
                return LABEL2ID["CROWD"]
            if any(w in text_lower for w in ["injur", "hospital", "ambulance", "medical", "hurt"]):
                return LABEL2ID["MEDICAL"]
            return LABEL2ID["ACCIDENT"]   # default disaster → accident

        hf_df["label"] = hf_df.apply(map_label, axis=1)
        hf_df["label_name"] = hf_df["label"].map(ID2LABEL)
        hf_df = hf_df[["text", "label", "label_name"]].dropna()

        combined = pd.concat([df, hf_df], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.success(f"Augmented: {len(df)} synthetic + {len(hf_df)} HuggingFace = {len(combined)} total")
        return combined

    except Exception as e:
        logger.warning(f"Augmentation parsing failed ({e}). Using synthetic only.")
        return df


def split_and_save(df: pd.DataFrame):
    """Splits into train/val/test (70/15/15) and saves CSV files."""
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv",   index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    logger.success(f"Saved → train:{len(train_df)} | val:{len(val_df)} | test:{len(test_df)}")
    return train_df, val_df, test_df


def print_distribution(df: pd.DataFrame, title: str = "Dataset"):
    table = Table(title=f"📊 {title} Distribution", border_style="dim")
    table.add_column("Label",   style="cyan")
    table.add_column("Count",   justify="right")
    table.add_column("Percent", justify="right")
    counts = df["label_name"].value_counts()
    for label, count in counts.items():
        pct = count / len(df) * 100
        table.add_row(label, str(count), f"{pct:.1f}%")
    table.add_row("TOTAL", str(len(df)), "100%", style="bold")
    console.print(table)


if __name__ == "__main__":
    logger.info("Building NLP training dataset...")

    # 1. Build from synthetic
    df = build_dataframe()
    logger.info(f"Synthetic data: {len(df)} sentences")

    # 2. Try to augment with HuggingFace
    df = augment_with_kaggle(df)

    # 3. Print distribution
    print_distribution(df, "Full Dataset")

    # 4. Split and save
    train_df, val_df, test_df = split_and_save(df)
    print_distribution(train_df, "Train Split")

    logger.success(f"Dataset ready in {DATA_DIR}")
    logger.info("Next: run  python scripts/train_nlp.py")