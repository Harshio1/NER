"""Template-based data augmentation by entity substitution.

Why: 10 sentences are far too few to train any sequence model. We keep the 10 real
recordings as the gold core and add synthetic sentences built from entity pools.
Train and dev pools are DISJOINT (dev measures generalisation to unseen entities) and
neither pool contains any entity used in the real test recordings.
"""
import random
from common import parse_markup

POOLS = {
 "PER": {
  "train": ["Anita Desai", "Karthik Subramanian", "Meena", "Suresh Raina", "Lakshmi", "Ravi Shankar",
            "Divya Menon", "Arun", "Kavitha", "Mohan Das", "Sachin Tendulkar", "Rohit Sharma", "Nisha",
            "Vijay Sethupathi", "Harini", "Satya Nadella", "Ashwin", "Bhavana", "Murali Krishnan",
            "Shruti Haasan", "Elon Musk", "Isha", "Gokul", "Sneha Reddy", "Sundar Pichai", "Priya", "Arjun"],
  "dev":   ["Manoj Bajpayee", "Pooja", "Dinesh Karthik", "Ananya Iyer", "Faisal", "Swathi", "Tarun",
            "Nithya Menen", "Kamal Haasan", "Zoya"]},
 "ORG": {
  "train": ["Zoho", "Flipkart", "Reliance Industries", "Hindustan Unilever", "Apple", "Tesla", "Indian Railways",
            "State Bank of India", "Larsen and Toubro", "Ashok Leyland", "Cognizant",
            "Bharat Petroleum", "IIT Madras", "Madras University", "Delhi University", "Hyundai Motor",
            "Samsung", "Paytm", "Swiggy", "Ola Electric", "Indian Oil Corporation", "Google", "Infosys"],
  "dev":   ["Zomato", "HCL Technologies", "Tech Mahindra", "PSG College", "NIT Trichy", "Bajaj Auto",
            "Air India", "Titan Company"]},
 "LOC": {
  "train": ["Delhi", "Salem", "Pune", "Trichy", "Tirunelveli", "Kolkata", "Goa", "Ooty", "Kanyakumari",
            "Mysuru", "Jaipur", "Vellore", "Thanjavur", "Puducherry", "Kerala", "Tamil Nadu", "Singapore",
            "London", "Dubai", "New York", "Tokyo", "Paris", "Andhra Pradesh", "Marina Beach", "Mumbai", "Chennai"],
  "dev":   ["Lucknow", "Hampi", "Sydney", "San Francisco", "Karnataka", "Nilgiris", "Kodaikanal", "Berlin"]},
}

TEMPLATES = [
 "{P} works at {O} in {L}.", "{P} was born in {L}.", "{O} announced a new plant in {L}.",
 "I met {P} at {L} last week.", "{P} joined {O} after studying at {O2}.",
 "We travelled from {L} to {L2} by train.", "{P} and {P2} visited {L} together.",
 "The {O} headquarters is located in {L}.", "{O} hired {P} as the new manager.",
 "My friend {P} lives near {L}.", "{P} said that {O} will open an office in {L}.",
 "She studied at {O} before moving to {L}.", "{O} and {O2} signed a deal in {L}.",
 "Yesterday {P} flew from {L} to {L2}.", "{P} is a professor at {O}.",
 "The conference in {L} was attended by {P} from {O}.", "{P} plays cricket for {O}.",
 "{O} is based in {L}.", "{P} will travel to {L} tomorrow.", "Please ask {P} about the meeting at {O}.",
 "On Monday {P} met {P2} in {L}.", "{P} visited {L} in January with {P2}.",
 "{O} shares rose on Friday after {P} spoke in {L}.", "The meeting with {O} is on Tuesday.",
 "{P} told officials from {O} about the trip to {L}.",
]
DISTRACTORS = [
 "The weather was very pleasant today.", "I am recording this sentence for my lab assignment.",
 "We had lunch at home on Sunday.", "The train was late by twenty minutes.",
 "Please close the window and switch off the fan.",
]
TAG = {"{P}": "PER", "{P2}": "PER", "{O}": "ORG", "{O2}": "ORG", "{L}": "LOC", "{L2}": "LOC"}


def generate(n, split="train", seed=0):
    rnd = random.Random(seed)
    out = []
    for k in range(n):
        if rnd.random() < 0.04:
            marked = rnd.choice(DISTRACTORS)
        else:
            marked = rnd.choice(TEMPLATES)
            used = {}
            for slot, typ in TAG.items():
                if slot in marked:
                    pool = POOLS[typ][split]
                    taken = {v for s, v in used.items() if TAG[s] == typ}
                    choice = rnd.choice([e for e in pool if e not in taken])
                    used[slot] = choice
                    marked = marked.replace(slot, f"[{choice}|{typ}]")
        toks, tags = parse_markup(marked)
        out.append({"id": f"{split[0].upper()}{k:04d}", "audio": "", "tokens": toks, "tags": tags})
    return out
