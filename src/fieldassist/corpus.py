"""Synthetic service documentation and golden question set.

Six product lines (four rotary screw compressors, two refrigerated dryers), one service manual
each, and eight service bulletins that reissue a manual section with one value changed. The
golden set is 300 questions across seven slices, built from the same facts the documents state,
so every expected answer is checkable. Everything is generated from a fixed seed; the committed
files under data/ are the output of `fieldassist seed` and a test keeps them in step.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SEED = 20260114
GOLDEN_SET_SIZE = 300

COMPRESSORS = {"KR-55": 55, "KR-75": 75, "KR-110": 110, "KR-160": 160}
DRYERS = {"TD-40": 40, "TD-75": 75}

COMPRESSOR_TORQUE = {"airend coupling bolts": (80, 140), "motor mount bolts": (60, 120),
                     "discharge flange bolts": (40, 90), "oil filter housing": (25, 50), "cooler fan hub": (30, 60)}
DRYER_TORQUE = {"condenser fan hub": (8, 20), "refrigerant service valve caps": (10, 20),
                "drain valve body": (20, 40), "heat exchanger mounting bolts": (25, 45)}

COMPRESSOR_FAULTS = [
    ("high discharge temperature", "the airend discharge passed 110 °C",
     "check the oil level and clean the cooler fins before restarting"),
    ("low oil pressure", "oil pressure fell below 1.5 bar while loaded",
     "check the oil level and the oil filter element"),
    ("motor overload", "the main motor overload relay tripped",
     "measure the motor current on all three phases and check the supply voltage"),
    ("phase loss", "one supply phase dropped out",
     "check the incoming fuses and the phase sequence relay"),
    ("separator differential pressure high", "pressure drop across the separator passed 1 bar",
     "replace the separator element"),
    ("inlet valve fails to open", "the inlet valve stayed closed on a load command",
     "check the solenoid coil and the control air line"),
    ("discharge temperature sensor open circuit", "the controller reads no signal from the sensor",
     "check the sensor plug and replace the sensor if its resistance is out of range"),
    ("pressure transducer out of range", "the line pressure signal is outside 4 to 20 mA",
     "check the transducer wiring and replace the transducer"),
    ("emergency stop pressed", "the emergency stop circuit is open",
     "release the button, find out why it was pressed and reset at the controller"),
]
DRYER_FAULTS = [
    ("high pressure dew point", "the dew point stayed above 10 °C for 15 minutes",
     "check the condenser for dirt and the refrigerant pressures"),
    ("condenser fan failure", "the fan motor drew no current when called",
     "check the fan capacitor and the fan motor windings"),
    ("low refrigerant pressure", "suction pressure fell below the cut-out setting",
     "leak-test the refrigerant circuit before recharging"),
    ("high refrigerant pressure", "discharge pressure passed the high-pressure switch setting",
     "clean the condenser and check that the fan runs"),
    ("drain valve blocked", "no condensate discharge was detected for 30 minutes",
     "clean the drain valve strainer and fit the service kit if it is worn"),
    ("ambient temperature high", "ambient air passed 45 °C at the condenser inlet",
     "improve ventilation around the dryer"),
]

COMPRESSOR_INTERVALS = [
    ("oil change", "change the oil", [2000, 4000, 6000]),
    ("air filter element", "replace the air filter element", [1000, 2000]),
    ("oil filter element", "replace the oil filter element", [2000, 4000]),
    ("separator element", "replace the separator element", [4000, 6000, 8000]),
    ("drive belt inspection", "inspect the drive belt", [1000, 2000]),
]
DRYER_INTERVALS = [
    ("condenser fin cleaning", "clean the condenser fins", [1000, 2000]),
    ("drain valve service kit", "fit the drain valve service kit", [4000, 8000]),
    ("pre-filter element", "replace the pre-filter element", [2000, 4000]),
    ("refrigerant leak check", "leak-check the refrigerant circuit", [4000, 8000]),
]

COMPRESSOR_PARTS = ["oil filter element", "air filter element", "separator element", "drive belt",
                    "discharge temperature sensor"]
DRYER_PARTS = ["drain valve service kit", "condenser fan motor", "pre-filter element", "dew point sensor"]

SECTION_TITLES = {
    "specifications": "Specifications",
    "torque-values": "Torque values",
    "fault-codes": "Fault codes",
    "maintenance-intervals": "Maintenance intervals",
    "parts": "Parts",
    "procedures": "Procedures",
}


@dataclass
class Line:
    name: str
    kind: str  # compressor | dryer
    specs: dict[str, str] = field(default_factory=dict)
    torque: dict[str, int] = field(default_factory=dict)
    faults: list[tuple[str, str, str, str]] = field(default_factory=list)  # code, name, cause, action
    intervals: dict[str, int] = field(default_factory=dict)
    parts: dict[str, str] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return "MAN-" + self.name.replace("-", "")


@dataclass
class Bulletin:
    doc_id: str
    line: str
    section: str
    item: str
    old: str
    new: str
    reason: str
    published: str


def _fmt_hours(h: int) -> str:
    return f"{h:,} hours"


def build_lines(rng: random.Random) -> dict[str, Line]:
    lines: dict[str, Line] = {}
    codes = iter(rng.sample(range(1000, 9999), 60))
    part_numbers = iter(rng.sample(range(1000, 9999), 60))
    family = rng.randint(4100, 4899)

    def next_pn() -> str:
        return f"{family}-{next(part_numbers):04d}"

    for name, kw in COMPRESSORS.items():
        ln = Line(name, "compressor")
        ln.specs = {
            "rated power": f"{kw} kW",
            "maximum working pressure": f"{rng.choice([10, 13])} bar",
            "free air delivery": f"{kw * 0.168 + rng.uniform(-0.6, 0.6):.1f} m³/min",
            "oil capacity": f"{int(kw * 0.42 + rng.randint(4, 12))} L",
            "shipping weight": f"{int(kw * 14 + rng.randint(200, 400)) // 10 * 10:,} kg",
        }
        ln.torque = {item: rng.choice(range(lo, hi + 1, 5)) for item, (lo, hi) in COMPRESSOR_TORQUE.items()}
        ln.faults = [(f"F{next(codes)}", n, c, a) for n, c, a in COMPRESSOR_FAULTS]
        ln.intervals = {item: rng.choice(opts) for item, _, opts in COMPRESSOR_INTERVALS}
        ln.parts = {p: next_pn() for p in COMPRESSOR_PARTS}
        lines[name] = ln
    for name, flow in DRYERS.items():
        ln = Line(name, "dryer")
        ln.specs = {
            "rated flow": f"{flow * 0.28:.1f} m³/min",
            "pressure dew point": f"{rng.choice([3, 5])} °C",
            "refrigerant charge": f"{flow * 0.021 + rng.uniform(0.1, 0.3):.2f} kg of R-513A",
            "maximum inlet temperature": f"{rng.choice([55, 60, 65])} °C",
            "shipping weight": f"{int(flow * 4 + rng.randint(60, 120)) // 5 * 5:,} kg",
        }
        ln.torque = {item: rng.choice(range(lo, hi + 1, 2)) for item, (lo, hi) in DRYER_TORQUE.items()}
        ln.faults = [(f"F{next(codes)}", n, c, a) for n, c, a in DRYER_FAULTS]
        ln.intervals = {item: rng.choice(opts) for item, _, opts in DRYER_INTERVALS}
        ln.parts = {p: next_pn() for p in DRYER_PARTS}
        lines[name] = ln
    lines["_pn"] = Line("_pn", "_")  # placeholder so bulletins can draw replacement part numbers
    lines["_pn"].parts = {"next": next_pn(), "next2": next_pn(), "next3": next_pn()}
    return lines


BULLETIN_PLAN = [
    ("KR-160", "torque-values", "airend coupling bolts", "coupling bolts loosening after the first 500 hours"),
    ("KR-110", "maintenance-intervals", "separator element", "separator elements clogging early in dusty sites"),
    ("KR-75", "parts", "oil filter element", "a supplier change for the oil filter element"),
    ("KR-55", "torque-values", "motor mount bolts", "motor mounts cracking on units with high vibration"),
    ("TD-40", "maintenance-intervals", "condenser fin cleaning", "high dew point alarms in summer"),
    ("TD-75", "parts", "drain valve service kit", "a redesigned drain valve seal"),
    ("KR-160", "maintenance-intervals", "oil change", "oil analysis showing early oxidation at high load"),
    ("TD-40", "torque-values", "refrigerant service valve caps", "slow refrigerant leaks at the valve caps"),
]


def build_bulletins(lines: dict[str, Line], rng: random.Random) -> list[Bulletin]:
    out: list[Bulletin] = []
    spare = iter(lines["_pn"].parts.values())
    for i, (line_name, section, item, reason) in enumerate(BULLETIN_PLAN):
        ln = lines[line_name]
        if section == "torque-values":
            old = ln.torque[item]
            new = old + rng.choice([5, 10]) if ln.kind == "compressor" else old + 4
            old_s, new_s = f"{old} Nm", f"{new} Nm"
        elif section == "maintenance-intervals":
            old = ln.intervals[item]
            new = old // 2
            old_s, new_s = _fmt_hours(old), _fmt_hours(new)
        else:
            old_s, new_s = ln.parts[item], next(spare)
        out.append(Bulletin(
            doc_id=f"SB-2026-{(i + 1) * 3 + 1:03d}", line=line_name, section=section, item=item,
            old=old_s, new=new_s, reason=reason, published=f"2026-{(i % 9) + 1:02d}-{10 + i:02d}",
        ))
    return out


# ----------------------------------------------------------------------------------------------
# Document text


def _section_text(ln: Line, section: str, override: tuple[str, str] | None = None) -> str:
    """Body of one manual section. `override` = (item, value) when a bulletin reissues it."""
    n = ln.name
    if section == "specifications":
        paras = [f"The {n} {k} is {v}." for k, v in ln.specs.items()]
        paras.append(f"All {n} figures are at 20 °C ambient and sea level unless a table says otherwise.")
        return "\n\n".join(paras)
    if section == "torque-values":
        vals = dict(ln.torque)
        if override:
            vals[override[0]] = int(override[1].split()[0])
        paras = [f"{item[0].upper() + item[1:]} on the {n}: tighten to {v} Nm in a cross pattern, "
                 f"then check again after the first 50 running hours." for item, v in vals.items()]
        paras.append(f"Use a calibrated torque wrench and clean, dry threads on the {n} unless a value "
                     f"above says lubricated.")
        return "\n\n".join(paras)
    if section == "fault-codes":
        return "\n\n".join(
            f"{code} on the {n}, {name}: {cause}. Action: {action}." for code, name, cause, action in ln.faults)
    if section == "maintenance-intervals":
        vals = dict(ln.intervals)
        if override:
            vals[override[0]] = int(override[1].split()[0].replace(",", ""))
        verbs = {i: v for i, v, _ in COMPRESSOR_INTERVALS + DRYER_INTERVALS}
        paras = [f"On the {n}, {verbs[item]} every {_fmt_hours(h)} of running time or once a year, "
                 f"whichever comes first." for item, h in vals.items()]
        paras.append(f"Shorten every {n} interval by half on sites with high dust or ambient air above 40 °C.")
        return "\n\n".join(paras)
    if section == "parts":
        vals = dict(ln.parts)
        paras = []
        for item, pn in vals.items():
            if override and override[0] == item:
                paras.append(f"{item[0].upper() + item[1:]} for the {n}: part {override[1]}, which replaces "
                             f"part {pn}.")
            else:
                paras.append(f"{item[0].upper() + item[1:]} for the {n}: part {pn}.")
        paras.append(f"Order {n} parts through the service portal with the unit serial number.")
        return "\n\n".join(paras)
    if section == "procedures":
        if ln.kind == "compressor":
            return (
                f"Before any work on the {n}, isolate the main supply, lock it out and vent the separator "
                f"tank until the gauge reads zero. Wait ten minutes for the oil to cool below 60 °C.\n\n"
                f"To replace the separator element on the {n}, remove the tank cover bolts in sequence, lift "
                f"out the old element with its gaskets, fit the new element with new gaskets and refit the "
                f"cover. Run the unit loaded for 15 minutes and check for leaks.\n\n"
                f"To change the oil on the {n}, drain it warm through the drain valve into a container, replace "
                f"the oil filter element, refill to the middle of the sight glass and run the unit for five "
                f"minutes before topping up."
            )
        return (
            f"Before any work on the {n}, isolate the supply and depressurize the air circuit. Refrigerant work "
            f"is for certified technicians only.\n\n"
            f"To clean the condenser on the {n}, remove the front panel, blow the fins clear from the inside "
            f"out with dry compressed air below 6 bar and check the fan blades for damage.\n\n"
            f"To service the drain valve on the {n}, close the isolation valve, release pressure with the test "
            f"button, remove the strainer and fit the parts from the service kit."
        )
    raise ValueError(section)


def _front_matter(meta: dict) -> str:
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n"


def manual_markdown(ln: Line) -> str:
    meta = {"doc_id": ln.doc_id, "kind": "manual", "title": f"{ln.name} Service Manual", "revision": "C",
            "product_line": ln.name, "published": "2025-03-14", "supersedes": []}
    parts = [_front_matter(meta), f"# {ln.name} Service Manual\n"]
    for sid, title in SECTION_TITLES.items():
        parts.append(f"## {title} {{#{sid}}}\n\n{_section_text(ln, sid)}\n")
    return "\n".join(parts)


def bulletin_markdown(b: Bulletin, lines: dict[str, Line]) -> str:
    ln = lines[b.line]
    noun = {"torque-values": "torque", "maintenance-intervals": "interval", "parts": "part number"}[b.section]
    title = f"Service Bulletin {b.doc_id}: {b.line} {b.item} {noun}"
    meta = {"doc_id": b.doc_id, "kind": "bulletin", "title": title, "revision": "A", "product_line": b.line,
            "published": b.published, "supersedes": [{"doc_id": ln.doc_id, "section": b.section}]}
    sid = b.section
    body = _section_text(ln, sid, override=(b.item, b.new))
    return "\n".join([
        _front_matter(meta),
        f"# {title}\n",
        f"## Reason {{#reason}}\n\nThis bulletin follows {b.reason}. It changes the {b.item} {noun} on the "
        f"{b.line} from {b.old} to {b.new}.\n",
        f"## {SECTION_TITLES[sid]} (reissued) {{#{sid}}}\n\n{body}\n",
        f"## Applies to {{#applies-to}}\n\nThis bulletin supersedes the {SECTION_TITLES[sid]} section of the "
        f"{b.line} Service Manual, revision C, for every {b.line} unit in the field.\n",
    ])


# ----------------------------------------------------------------------------------------------
# Golden set


SPEC_Q = {
    "maximum working pressure": ["What is the maximum working pressure of the {n}?",
                                 "How many bar can the {n} run at, maximum working pressure?"],
    "free air delivery": ["What is the free air delivery of the {n}?",
                          "How much free air delivery does a {n} give?"],
    "oil capacity": ["What is the oil capacity of the {n}?", "{n} oil capacity?"],
    "shipping weight": ["What is the shipping weight of the {n}?", "How heavy is a {n}, shipping weight?"],
    "rated flow": ["What is the rated flow of the {n} dryer?", "{n} rated flow?"],
    "pressure dew point": ["What pressure dew point does the {n} reach?",
                           "What is the {n} pressure dew point?"],
    "refrigerant charge": ["What is the refrigerant charge of the {n}?",
                           "How much refrigerant charge goes in a {n}?"],
    "maximum inlet temperature": ["What is the maximum inlet temperature for the {n}?",
                                  "{n} maximum inlet temperature?"],
}
TORQUE_Q = ["What torque do the {item} on the {n} take?", "What should I tighten the {item} to on a {n}?",
            "{n}: torque for the {item}?", "Torque value for the {n} {item}?"]
INTERVAL_Q = ["How often should I {verb} on the {n}?", "What is the interval to {verb} on a {n}?",
              "After how many running hours do you {verb} on the {n}?"]
FAULT_Q = ["What does fault {code} mean?", "The controller is showing {code}. What does that mean?",
           "A customer reports fault code {code}, what is it?", "What is fault {code}?"]
PART_Q = ["What is part {pn}?", "What does part number {pn} belong to?", "Which unit is part {pn} for?"]
PART_LOOKUP_Q = ["What is the part number for the {item} on the {n}?", "Part number for the {n} {item}?"]
OUT_OF_SCOPE = [
    "How do I reset the thermostat in the break room?",
    "What is the warranty period on a new forklift battery?",
    "How do I submit a time-off request?",
    "Which diesel generator should the plant buy next year?",
    "What is the guest wifi password at the head office?",
    "How do I book a meeting room for Thursday?",
    "What is the mileage allowance for a personal car?",
    "Who approves overtime for the night shift?",
    "How do I calibrate the paint booth humidity sensor?",
    "What is the lead time on a new loading dock door?",
    "How do I expense a hotel booking?",
    "What size are the pallets in the shipping area?",
]
UNKNOWN_MODELS = ["KR-300", "KR-90", "TD-150", "TD-20"]


def build_golden_set(lines: dict[str, Line], bulletins: list[Bulletin], rng: random.Random) -> list[dict]:
    real = [ln for k, ln in lines.items() if not k.startswith("_")]
    superseded = {(b.line, b.section, b.item): b for b in bulletins}
    pools: dict[str, list[dict]] = {k: [] for k in
                                    ["spec", "torque", "interval", "fault_code", "part_number", "superseded",
                                     "out_of_scope"]}

    def q(slice_, text, facts, doc, section, answerable=True):
        pools[slice_].append({"slice": slice_, "question": text, "key_facts": facts, "expected_doc": doc,
                              "expected_section": section, "answerable": answerable})

    verbs = {i: v for i, v, _ in COMPRESSOR_INTERVALS + DRYER_INTERVALS}
    # The current text of a section a bulletin reissued lives in the bulletin, unchanged items included.
    owner = {(b.line, b.section): b.doc_id for b in bulletins}
    for ln in real:
        def src(section: str, line=ln) -> str:
            return owner.get((line.name, section), line.doc_id)

        for attr, value in ln.specs.items():
            for t in SPEC_Q.get(attr, []):
                q("spec", t.format(n=ln.name), [value.split(" of ")[0], ln.name], src("specifications"),
                  "specifications")
        for item, nm in ln.torque.items():
            b = superseded.get((ln.name, "torque-values", item))
            for t in TORQUE_Q:
                if b:
                    q("superseded", t.format(item=item, n=ln.name), [b.new, ln.name], b.doc_id, "torque-values")
                else:
                    q("torque", t.format(item=item, n=ln.name), [f"{nm} Nm", ln.name], src("torque-values"),
                      "torque-values")
        for item, h in ln.intervals.items():
            b = superseded.get((ln.name, "maintenance-intervals", item))
            for t in INTERVAL_Q:
                text = t.format(verb=verbs[item], n=ln.name)
                if b:
                    q("superseded", text, [b.new, ln.name], b.doc_id, "maintenance-intervals")
                else:
                    q("interval", text, [_fmt_hours(h), ln.name], src("maintenance-intervals"),
                      "maintenance-intervals")
        for code, name, _, _ in ln.faults:
            for t in FAULT_Q:
                q("fault_code", t.format(code=code), [name, ln.name], src("fault-codes"), "fault-codes")
        for item, pn in ln.parts.items():
            b = superseded.get((ln.name, "parts", item))
            if b:
                for t in PART_LOOKUP_Q:
                    q("superseded", t.format(item=item, n=ln.name), [b.new, ln.name], b.doc_id, "parts")
                continue
            for t in PART_Q:
                q("part_number", t.format(pn=pn), [item, ln.name], src("parts"), "parts")
    for text in OUT_OF_SCOPE:
        q("out_of_scope", text, [], None, None, answerable=False)
    for model in UNKNOWN_MODELS:
        q("out_of_scope", f"What torque do the motor mount bolts on the {model} take?", [], None, None, False)
        q("out_of_scope", f"How often should I change the oil on the {model}?", [], None, None, False)
        q("out_of_scope", f"What does the {model} weigh, shipping weight?", [], None, None, False)

    quota = {"spec": 45, "torque": 60, "interval": 45, "fault_code": 60, "part_number": 42, "superseded": 24,
             "out_of_scope": 24}
    assert sum(quota.values()) == GOLDEN_SET_SIZE
    chosen: list[dict] = []
    for slice_, n in quota.items():
        pool = pools[slice_]
        if len(pool) < n:
            raise ValueError(f"slice {slice_} has {len(pool)} candidates, needs {n}")
        chosen.extend(rng.sample(pool, n))
    for i, item in enumerate(chosen, 1):
        item["id"] = f"q{i:03d}"
    return [{"id": c["id"], **{k: v for k, v in c.items() if k != "id"}} for c in chosen]


# ----------------------------------------------------------------------------------------------


def generate() -> tuple[dict[str, str], list[dict]]:
    """Return ({relative path: markdown}, golden set)."""
    rng = random.Random(SEED)
    lines = build_lines(rng)
    bulletins = build_bulletins(lines, rng)
    files: dict[str, str] = {}
    for name, ln in lines.items():
        if name.startswith("_"):
            continue
        files[f"corpus/manuals/{ln.doc_id}.md"] = manual_markdown(ln)
    for b in bulletins:
        files[f"corpus/bulletins/{b.doc_id}.md"] = bulletin_markdown(b, lines)
    golden = build_golden_set(lines, bulletins, rng)
    return files, golden


def write(data_dir: Path) -> dict[str, int]:
    files, golden = generate()
    for rel, text in files.items():
        path = data_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    gs = data_dir / "golden_set.jsonl"
    gs.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in golden), encoding="utf-8",
                  newline="\n")
    return {"documents": len(files), "questions": len(golden)}


def load_golden_set(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
