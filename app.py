"""
╔══════════════════════════════════════════════════════════════════╗
║        EnsemblePhish v4 — Professional Edition                   ║
║                                                                  ║
║  Pages: Home · URL Detection · Bulk Scan · Dataset · About       ║
║  Bulk Scan: paste URLs, upload .txt, or upload PDF               ║
║  Clean, premium dark cyber UI                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────
import re
import ssl
import socket
import urllib.parse
import warnings
from io import StringIO, BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  SECTION A — CONSTANTS
# ═══════════════════════════════════════════════════════════════

ALL_FEATURE_COLS = [
    "having_IPhaving_IP_Address", "URLURL_Length", "Shortining_Service",
    "having_At_Symbol", "double_slash_redirecting", "Prefix_Suffix",
    "having_Sub_Domain", "SSLfinal_State", "Domain_registeration_length",
    "Favicon", "port", "HTTPS_token", "Request_URL", "URL_of_Anchor",
    "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe",
    "age_of_domain", "DNSRecord", "web_traffic", "Page_Rank",
    "Google_Index", "Links_pointing_to_page", "Statistical_report",
]

TRUSTED_DOMAINS = {
    "google.com", "gmail.com", "youtube.com", "microsoft.com",
    "outlook.com", "office.com", "live.com", "apple.com", "icloud.com",
    "amazon.com", "amazon.in", "github.com", "gitlab.com",
    "wikipedia.org", "stackoverflow.com", "linkedin.com",
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "whatsapp.com", "openai.com", "chatgpt.com", "anthropic.com",
    "claude.ai", "netflix.com", "spotify.com", "paypal.com",
    "cloudflare.com", "reddit.com", "zoom.us",
}

PHISH_KEYWORDS = [
    "login", "verify", "bank", "secure", "update", "account", "confirm",
    "signin", "password", "credential", "webscr", "ebayisapi",
    "suspend", "unlock", "validate",
]


# ═══════════════════════════════════════════════════════════════
#  SECTION B — URL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def feat_ip_address(url: str) -> int:
    pattern = r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])"
    return -1 if re.search(pattern, url) else 1

def feat_url_length(url: str) -> int:
    n = len(url)
    if n < 54:   return 1
    if n <= 75:  return 0
    return -1

def feat_shortening_service(url: str) -> int:
    pattern = r"(bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|t\.co|is\.gd|buff\.ly|rebrand\.ly|cutt\.ly|short\.link)"
    return -1 if re.search(pattern, url, re.IGNORECASE) else 1

def feat_at_symbol(url: str) -> int:
    return -1 if "@" in url else 1

def feat_double_slash(url: str) -> int:
    return -1 if url.find("//", 7) > 0 else 1

def feat_prefix_suffix(url: str) -> int:
    domain = re.sub(r"https?://", "", url).split("/")[0].split(":")[0]
    return -1 if "-" in domain else 1

def feat_subdomains(url: str) -> int:
    domain = re.sub(r"https?://", "", url).split("/")[0].split(":")[0].lower()
    if domain.startswith("www."):
        domain = domain[4:]
    dot_count = domain.count(".")
    if dot_count <= 1: return 1
    if dot_count == 2: return 0
    return -1

def feat_ssl_state(url: str) -> int:
    if not url.startswith("https"):
        return -1
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.split(":")[0]
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as sock:
            sock.settimeout(3)
            sock.connect((domain, 443))
            return 1
    except ssl.SSLCertVerificationError:
        return -1
    except Exception:
        return 0

def feat_https_token(url: str) -> int:
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    return -1 if "https" in domain else 1

def extract_url_features(url: str) -> dict:
    return {
        "having_IPhaving_IP_Address": feat_ip_address(url),
        "URLURL_Length":               feat_url_length(url),
        "Shortining_Service":          feat_shortening_service(url),
        "having_At_Symbol":            feat_at_symbol(url),
        "double_slash_redirecting":    feat_double_slash(url),
        "Prefix_Suffix":               feat_prefix_suffix(url),
        "having_Sub_Domain":           feat_subdomains(url),
        "SSLfinal_State":              feat_ssl_state(url),
        "HTTPS_token":                 feat_https_token(url),
    }


# ═══════════════════════════════════════════════════════════════
#  SECTION C — RULE-BASED SCORING
# ═══════════════════════════════════════════════════════════════

def get_base_domain(url: str) -> str:
    domain = re.sub(r"https?://", "", url).split("/")[0].split(":")[0].lower()
    if domain.startswith("www."):
        domain = domain[4:]
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain

def rule_based_score(url: str, url_feats: dict) -> tuple:
    score = 0
    reasons = []

    if url_feats["having_IPhaving_IP_Address"] == -1:
        score += 40
        reasons.append(("Uses raw IP address instead of a domain name", True))
    else:
        reasons.append(("Uses a domain name (not a raw IP)", False))

    if "@" in url:
        score += 40
        reasons.append(("Contains @ symbol — forces browser to ignore the left part", True))
    else:
        reasons.append(("No @ symbol in URL", False))

    if url_feats["SSLfinal_State"] == -1:
        score += 30
        reasons.append(("No HTTPS — connection is unencrypted and unverified", True))
    elif url_feats["SSLfinal_State"] == 1:
        score -= 10
        reasons.append(("Valid HTTPS certificate present", False))
    else:
        reasons.append(("HTTPS present but certificate not verified", False))

    if url_feats["Shortining_Service"] == -1:
        score += 20
        reasons.append(("Uses a URL shortening service (hides real destination)", True))
    else:
        reasons.append(("No URL shortening service detected", False))

    if url_feats["having_Sub_Domain"] == -1:
        score += 15
        reasons.append(("Too many subdomains (common phishing trick)", True))
    elif url_feats["having_Sub_Domain"] == 1:
        reasons.append(("Normal subdomain structure", False))

    if url_feats["Prefix_Suffix"] == -1:
        score += 10
        reasons.append(("Domain name contains a hyphen (e.g. paypal-secure.com)", True))
    else:
        reasons.append(("No hyphens in domain name", False))

    url_lower = url.lower()
    found_keywords = [kw for kw in PHISH_KEYWORDS if kw in url_lower]
    if len(found_keywords) >= 2:
        score += 20
        reasons.append((f"Multiple suspicious keywords found: {', '.join(found_keywords[:4])}", True))
    elif len(found_keywords) == 1:
        score += 8
        reasons.append((f"Suspicious keyword in URL: '{found_keywords[0]}'", True))
    else:
        reasons.append(("No suspicious keywords in URL", False))

    if url_feats["URLURL_Length"] == -1:
        score += 10
        reasons.append(("URL is unusually long (>75 characters)", True))
    elif url_feats["URLURL_Length"] == 1:
        reasons.append(("URL length is normal", False))

    score = max(0, min(100, score))
    return score, reasons


# ═══════════════════════════════════════════════════════════════
#  SECTION D — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def train_models(dataset_content=None):
    if dataset_content is not None:
        df = pd.read_csv(StringIO(dataset_content.decode("utf-8")))
        source_label = "uploaded custom dataset"
    else:
        df = pd.read_csv("dataset.csv")
        source_label = "default dataset"

    df.drop(columns=["index"], inplace=True, errors="ignore")
    df.fillna(df.median(numeric_only=True), inplace=True)

    label_col = None
    for col in df.columns:
        if col.lower() in ["result", "label", "phishing", "target", "class"]:
            label_col = col
            break
    if label_col is None:
        label_col = df.columns[-1]

    feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    if len(feat_cols) < 5:
        feat_cols = [c for c in df.columns if c != label_col]

    X = df[feat_cols]

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    classes = list(le.classes_)

    phishing_class_idx = 0
    if -1 in classes:
        phishing_class_idx = classes.index(-1)
    elif "phishing" in [str(c).lower() for c in classes]:
        phishing_class_idx = [str(c).lower() for c in classes].index("phishing")
    elif 1 in classes and -1 not in classes:
        phishing_class_idx = classes.index(1)

    legit_label = classes[1 - phishing_class_idx]
    legit_mask = df[label_col] == legit_label
    legit_medians = df.loc[legit_mask, feat_cols].median().to_dict()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))

    return {
        "rf": rf, "svm": svm, "mlp": mlp,
        "rf_acc": rf_acc, "svm_acc": svm_acc, "mlp_acc": mlp_acc,
        "feat_cols": feat_cols,
        "phishing_idx": phishing_class_idx,
        "legit_medians": legit_medians,
        "source": source_label,
        "n_samples": len(df),
        "n_features": len(feat_cols),
        "df": df,
        "label_col": label_col,
    }


# ═══════════════════════════════════════════════════════════════
#  SECTION E — ENSEMBLE PREDICTION
# ═══════════════════════════════════════════════════════════════

def build_feature_vector(url_feats: dict, m: dict) -> np.ndarray:
    vector = []
    for col in m["feat_cols"]:
        if col in url_feats:
            vector.append(url_feats[col])
        else:
            vector.append(m["legit_medians"][col])
    return np.array(vector).reshape(1, -1)

def ml_predict(url_feats: dict, m: dict) -> dict:
    arr = build_feature_vector(url_feats, m)
    pidx = m["phishing_idx"]
    votes, probas = {}, {}
    for name, model in [("Random Forest", m["rf"]), ("SVM", m["svm"]), ("Neural Net", m["mlp"])]:
        raw_pred = model.predict(arr)[0]
        is_phishing_vote = 1 if raw_pred == pidx else 0
        votes[name] = is_phishing_vote
        probas[name] = model.predict_proba(arr)[0][pidx]
    phish_votes = sum(votes.values())
    ml_is_phishing = phish_votes >= 2
    avg_proba = sum(probas.values()) / 3
    return {
        "is_phishing": ml_is_phishing,
        "avg_proba": avg_proba,
        "votes": votes,
        "probas": probas,
        "phish_votes": phish_votes,
    }

def combined_predict(url: str, url_feats: dict, m: dict) -> dict:
    base_domain = get_base_domain(url)
    if base_domain in TRUSTED_DOMAINS:
        return {
            "verdict": "safe", "label": "LEGITIMATE",
            "confidence": 0.0, "rule_score": 0,
            "override": "trusted_domain",
            "override_reason": f'"{base_domain}" is a globally trusted domain',
            "ml": None, "reasons": [],
        }
    rule_score, reasons = rule_based_score(url, url_feats)
    ml = ml_predict(url_feats, m)
    if rule_score >= 70:
        return {
            "verdict": "phishing", "label": "PHISHING DETECTED",
            "confidence": rule_score / 100, "rule_score": rule_score,
            "override": "rule", "override_reason": None,
            "ml": ml, "reasons": reasons,
        }
    blended = 0.6 * ml["avg_proba"] + 0.4 * (rule_score / 100)
    if blended >= 0.65:
        verdict, label = "phishing", "PHISHING DETECTED"
    elif blended >= 0.40:
        verdict, label = "suspicious", "SUSPICIOUS"
    else:
        verdict, label = "safe", "LEGITIMATE"
    return {
        "verdict": verdict, "label": label,
        "confidence": blended, "rule_score": rule_score,
        "override": None, "override_reason": None,
        "ml": ml, "reasons": reasons,
    }


# ═══════════════════════════════════════════════════════════════
#  SECTION E2 — PDF LINK EXTRACTION HELPER
# ═══════════════════════════════════════════════════════════════

def extract_urls_from_pdf(pdf_bytes: bytes) -> list:
    """Extract URLs from a PDF file. Uses PyMuPDF if available, falls back to regex."""
    urls = []
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            # Extract links from annotations
            for link in page.get_links():
                uri = link.get("uri", "")
                if uri and uri.startswith("http"):
                    urls.append(uri)
            # Also regex-scan the text for any http(s) URLs
            text = page.get_text()
            found = re.findall(r'https?://[^\s<>"\')\]]+', text)
            urls.extend(found)
        doc.close()
    except ImportError:
        # Fallback: regex scan the raw PDF bytes
        try:
            raw_text = pdf_bytes.decode("latin-1", errors="ignore")
        except Exception:
            raw_text = str(pdf_bytes)
        found = re.findall(r'https?://[^\s<>"\')\]]+', raw_text)
        urls.extend(found)

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        u_clean = u.rstrip("/.,;:!?")
        if u_clean not in seen:
            seen.add(u_clean)
            unique_urls.append(u_clean)
    return unique_urls


# ═══════════════════════════════════════════════════════════════
#  SECTION F — PAGE CONFIG & GLOBAL CSS
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EnsemblePhish — Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (premium dark cyber theme) ─────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0a0f1a !important;
    color: #c8d6e5 !important;
}

/* Hide sidebar */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Remove Streamlit header chrome */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: transparent !important;
    box-shadow: none !important;
}

[data-testid="stAppViewContainer"],
div.block-container,
section.main,
div[data-testid="stBlock"] {
    background: transparent !important;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(160deg, #0a0f1a 0%, #0d1525 40%, #0a1020 100%);
    color: #c8d6e5;
}

/* ── TOP NAV BAR ── */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(10, 15, 26, 0.97);
    border-bottom: 1px solid rgba(56, 189, 248, 0.08);
    padding: 0 32px;
    height: 64px;
    position: sticky;
    top: 0;
    z-index: 999;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    margin-bottom: 0;
}
.nav-brand {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    background-clip: text;
    letter-spacing: 2px;
    white-space: nowrap;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, rgba(56,189,248,0.05) 0%, rgba(129,140,248,0.05) 50%, rgba(56,189,248,0.03) 100%);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 20px;
    padding: 56px 48px 48px 48px;
    text-align: center;
    margin: 24px 0 20px 0;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -100px; right: -100px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -80px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(129,140,248,0.05) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Inter', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    margin: 0 0 12px 0;
}
.hero .tagline {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
    margin: 0 0 28px 0;
    letter-spacing: 0.5px;
}
.hero .badges {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 6px;
}
.badge-pill {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.75rem;
    font-weight: 500;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.3px;
}

/* ── STAT CARDS ── */
.stat-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56,189,248,0.08);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.stat-card:hover {
    border-color: rgba(56,189,248,0.2);
    box-shadow: 0 8px 30px rgba(56,189,248,0.06);
    transform: translateY(-2px);
}
.stat-card .stat-icon { font-size: 1.8rem; margin-bottom: 10px; }
.stat-card .stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-card .stat-lbl {
    font-size: 0.78rem;
    color: #475569;
    margin-top: 6px;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ── SECTION BOX ── */
.box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(56,189,248,0.08);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
}
.box-title {
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── RESULT BANNERS ── */
.res-phish {
    background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(239,68,68,0.05) 100%);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ef4444;
    letter-spacing: 3px;
    margin: 12px 0;
}
.res-warn {
    background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(245,158,11,0.05) 100%);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f59e0b;
    letter-spacing: 3px;
    margin: 12px 0;
}
.res-safe {
    background: linear-gradient(135deg, rgba(34,197,94,0.1) 0%, rgba(34,197,94,0.05) 100%);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #22c55e;
    letter-spacing: 3px;
    margin: 12px 0;
}

/* ── RISK BAR ── */
.risk-track {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(56,189,248,0.08);
    border-radius: 10px;
    height: 32px;
    overflow: hidden;
    margin: 10px 0 4px 0;
}
.risk-fill {
    height: 100%;
    border-radius: 10px;
    display: flex;
    align-items: center;
    padding-left: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    color: #fff;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── ACCENT BAR (accuracy) ── */
.acc-track {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(56,189,248,0.06);
    border-radius: 8px;
    height: 22px;
    overflow: hidden;
    margin: 6px 0 2px 0;
}
.acc-fill {
    height: 100%;
    border-radius: 8px;
    display: flex;
    align-items: center;
    padding-left: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
}

/* ── STEP CARDS ── */
.step-card {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(56,189,248,0.06);
    border-left: 3px solid #38bdf8;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 10px;
    transition: all 0.2s ease;
}
.step-card:hover {
    border-left-color: #818cf8;
    background: rgba(15,23,42,0.8);
}
.step-card h5 {
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0 0 8px 0;
    letter-spacing: 1px;
}
.step-card p {
    color: #64748b;
    font-size: 0.86rem;
    margin: 0;
    line-height: 1.65;
}

/* ── MODEL CARD ── */
.model-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(56,189,248,0.08);
    border-top: 3px solid;
    border-image: linear-gradient(135deg, #38bdf8, #818cf8) 1;
    border-radius: 12px;
    padding: 22px;
    height: 100%;
    transition: all 0.3s ease;
}
.model-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(56,189,248,0.08);
}
.model-card .model-icon { font-size: 2rem; margin-bottom: 10px; }
.model-card h5 {
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    font-size: 0.84rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin: 0 0 10px 0;
}
.model-card p { color: #64748b; font-size: 0.84rem; line-height: 1.65; margin: 0; }

/* ── BUTTONS ── */
.stButton > button,
button[kind="primary"],
button[kind="secondary"] {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 0 !important;
    width: 120%;
    min-height: 48px;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    letter-spacing: 0.5px;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(56,189,248,0.2);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.stButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(56,189,248,0.3) !important;
    background: linear-gradient(135deg, #60ccfa, #9ba3f8) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
main[class*="block-container"],
section[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.6);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(56,189,248,0.08);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #475569;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    padding: 10px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: rgba(56,189,248,0.1) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px;
}

/* ── Bulk scan result rows ── */
.bulk-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    transition: background 0.2s;
}
.bulk-row:hover {
    background: rgba(56,189,248,0.04);
}
.bulk-safe {
    background: rgba(34,197,94,0.06);
    border-left: 3px solid #22c55e;
}
.bulk-warn {
    background: rgba(245,158,11,0.06);
    border-left: 3px solid #f59e0b;
}
.bulk-phish {
    background: rgba(239,68,68,0.06);
    border-left: 3px solid #ef4444;
}
.bulk-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    min-width: 80px;
    text-align: center;
}
.tag-safe   { background: rgba(34,197,94,0.15); color: #22c55e; }
.tag-warn   { background: rgba(245,158,11,0.15); color: #f59e0b; }
.tag-phish  { background: rgba(239,68,68,0.15); color: #ef4444; }
.bulk-url   { color: #94a3b8; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bulk-score { color: #64748b; font-size: 0.75rem; min-width: 60px; text-align: right; }

/* ── Summary cards for bulk scan ── */
.summary-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(56,189,248,0.08);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.summary-card .s-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
}
.summary-card .s-lbl {
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 4px;
    color: #475569;
    letter-spacing: 0.3px;
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    padding: 28px;
    color: #1e293b;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
    border-top: 1px solid rgba(56,189,248,0.06);
    margin-top: 48px;
}
.divider { border-top: 1px solid rgba(56,189,248,0.06); margin: 24px 0; }

/* ── Input styling ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(56,189,248,0.12) !important;
    border-radius: 10px !important;
    color: #c8d6e5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(56,189,248,0.3) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.08) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}

/* ── Metric styling ── */
[data-testid="stMetric"] {
    background: rgba(15,23,42,0.5);
    border: 1px solid rgba(56,189,248,0.06);
    border-radius: 12px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SECTION G — SESSION STATE & TOP NAV
# ═══════════════════════════════════════════════════════════════

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "dataset_bytes" not in st.session_state:
    st.session_state.dataset_bytes = None

# ── Top Navigation Bar ────────────────────────────────────────
nav_cols = st.columns([2.2, 1, 1.2, 1.2, 1.2, 1])
with nav_cols[0]:
    st.markdown('<div class="nav-brand">🛡️ ENSEMBLEPHISH</div>', unsafe_allow_html=True)

pages = ["Home", "URL Detection", "Bulk Scan", "Dataset", "About"]
page_icons = ["🏠", "🔗", "📋", "📊", "ℹ️"]

for i, (col, pkey, icon) in enumerate(zip(nav_cols[1:], pages, page_icons)):
    with col:
        if st.button(f"{icon} {pkey}", key=f"nav_{pkey}"):
            st.session_state.page = pkey
            st.rerun()

page = st.session_state.page


# ═══════════════════════════════════════════════════════════════
#  SECTION H — HOME PAGE
# ═══════════════════════════════════════════════════════════════

if page == "Home":

    # Hero banner
    st.markdown("""
    <div class="hero">
      <h1>ENSEMBLEPHISH</h1>
      <p class="tagline">AI-Powered Phishing URL Detection — Rule Engine + ML Ensemble + Explainable Results</p>
      <div class="badges">
        <span class="badge-pill">3-Model Ensemble</span>
        <span class="badge-pill">Rule-Based Scoring</span>
        <span class="badge-pill">Bulk URL Scanning</span>
        <span class="badge-pill">PDF Link Extraction</span>
        <span class="badge-pill">96.7% Accuracy</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    cards = [
        ("🔬", "11,055", "URLs in Dataset"),
        ("🧠", "3", "ML Models"),
        ("⚡", "30", "Feature Dimensions"),
        ("🎯", "96.7%", "Ensemble Accuracy"),
    ]
    for col, (icon, val, lbl) in zip([sc1, sc2, sc3, sc4], cards):
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-icon">{icon}</div>
              <div class="stat-val">{val}</div>
              <div class="stat-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Architecture overview
    st.markdown("#### System Architecture")
    ar1, ar2 = st.columns(2)
    with ar1:
        st.markdown("""
        <div class="step-card">
          <h5>LAYER 1 · WHITELIST CHECK</h5>
          <p>Globally trusted domains (Google, Apple, GitHub, etc.) are instantly
          classified as safe — no ML needed. Eliminates false positives on major sites.</p>
        </div>
        <div class="step-card">
          <h5>LAYER 2 · RULE-BASED SCORING</h5>
          <p>Deterministic rules check for raw IP addresses, @ symbols, missing HTTPS,
          URL shorteners, suspicious keywords, and hyphenated domains. Score 0–100.</p>
        </div>
        """, unsafe_allow_html=True)
    with ar2:
        st.markdown("""
        <div class="step-card">
          <h5>LAYER 3 · ML ENSEMBLE</h5>
          <p>Random Forest + SVM (RBF) + MLP Neural Network vote on the URL.
          Results are blended with rule scores: 60% ML · 40% rules.</p>
        </div>
        <div class="step-card">
          <h5>LAYER 4 · BLENDED DECISION</h5>
          <p>Final score combines ML probability and rule score. Clear thresholds
          map to Safe, Suspicious, or Phishing verdicts with full explainability.</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SECTION I — URL DETECTION PAGE
# ═══════════════════════════════════════════════════════════════

elif page == "URL Detection":

    st.markdown("## 🔗 URL Detection")
    st.markdown("<p style='color:#64748b; margin-top:-8px;'>Paste any URL and let the ensemble analyse it in real time.</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models…"):
        try:
            m = train_models(st.session_state.dataset_bytes)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()

    # URL input
    st.markdown('<div class="box"><div class="box-title">TARGET URL</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "Enter the URL you want to check:",
        placeholder="https://example.com   or   http://192.168.1.1/login/verify/account",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    analyze_btn = st.button("🔍   ANALYZE URL")

    if analyze_btn:
        if not url_input.strip():
            st.warning("Please enter a URL first.")
        else:
            url = url_input.strip()
            if not url.startswith("http"):
                url = "http://" + url

            with st.spinner("Extracting features and running ensemble…"):
                url_feats = extract_url_features(url)
                result    = combined_predict(url, url_feats, m)

            # Verdict banner
            st.markdown("### Verdict")
            v = result["verdict"]
            css_cls = {"phishing": "res-phish", "suspicious": "res-warn", "safe": "res-safe"}.get(v, "res-safe")
            icon = {"phishing": "🚨", "suspicious": "⚠️", "safe": "✅"}.get(v, "✅")
            st.markdown(f'<div class="{css_cls}">{icon}  {result["label"]}</div>', unsafe_allow_html=True)

            # Override notices
            if result["override"] == "trusted_domain":
                st.success(f"🛡️ **Trusted domain bypass:** {result['override_reason']}")
            elif result["override"] == "rule":
                st.error("🚨 **Rule override active:** Strong phishing signals detected — ML vote overridden.")

            # ML vote cards
            if result["ml"]:
                ml = result["ml"]
                st.markdown("### Model Votes")
                c1, c2, c3 = st.columns(3)
                for col, (mname, vote) in zip([c1, c2, c3], ml["votes"].items()):
                    p = ml["probas"][mname]
                    if vote == 1:
                        col.error(f"**{mname}**\n\n🚨 Phishing\n\nConfidence: {p*100:.0f}%")
                    else:
                        col.success(f"**{mname}**\n\n✅ Legitimate\n\nConfidence: {(1-p)*100:.0f}%")
                st.caption(
                    f"Phishing votes: {ml['phish_votes']}/3  ·  "
                    f"Avg ML phishing prob: {ml['avg_proba']*100:.1f}%  ·  "
                    f"Rule score: {result['rule_score']}/100  ·  "
                    f"Blended score: {result['confidence']*100:.1f}%"
                )

            # Risk bar
            score_pct = int(result["confidence"] * 100)
            bar_color = "#ef4444" if score_pct >= 65 else ("#f59e0b" if score_pct >= 40 else "#22c55e")
            st.markdown("### Risk Score")
            st.markdown(f"""
            <div class="risk-track">
              <div class="risk-fill" style="width:{score_pct}%; background:{bar_color};">
                {score_pct} / 100
              </div>
            </div>
            <small style="color:#475569">0–39 = Safe &nbsp;|&nbsp; 40–64 = Suspicious &nbsp;|&nbsp; 65–100 = Phishing</small>
            """, unsafe_allow_html=True)

            # Extracted features
            st.markdown("### Extracted Features")
            val_map = {1: "✅ Safe", 0: "⚪ Neutral", -1: "🚨 Suspicious"}
            feat_labels = {
                "having_IPhaving_IP_Address": "IP Address in URL",
                "URLURL_Length":               "URL Length",
                "Shortining_Service":          "URL Shortener",
                "having_At_Symbol":            "@ Symbol",
                "double_slash_redirecting":    "Double Slash Redirect",
                "Prefix_Suffix":               "Hyphen in Domain",
                "having_Sub_Domain":           "Sub-Domain Count",
                "SSLfinal_State":              "HTTPS / SSL State",
                "HTTPS_token":                 "'https' in Domain Name",
            }
            feat_df = pd.DataFrame([
                {"Feature": feat_labels.get(k, k), "Raw Value": v, "Status": val_map.get(v, str(v))}
                for k, v in url_feats.items()
            ])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # Explanation panel
            st.markdown("### Why This Result?")
            bad  = [r for r, is_bad in result["reasons"] if is_bad]
            good = [r for r, is_bad in result["reasons"] if not is_bad]
            col_bad, col_good = st.columns(2)
            with col_bad:
                st.markdown('<div class="box"><div class="box-title">⚠️ SUSPICIOUS INDICATORS</div>', unsafe_allow_html=True)
                if bad:
                    for r in bad:
                        st.markdown(f"- 🚨 {r}")
                else:
                    st.markdown("- None found ✅")
                st.markdown("</div>", unsafe_allow_html=True)
            with col_good:
                st.markdown('<div class="box"><div class="box-title">✅ SAFE INDICATORS</div>', unsafe_allow_html=True)
                for r in good[:6]:
                    st.markdown(f"- ✅ {r}")
                st.markdown("</div>", unsafe_allow_html=True)

            # Final advice
            if v == "phishing":
                st.error("**Do NOT enter any personal information on this site.** This URL has multiple phishing indicators.")
            elif v == "suspicious":
                st.warning("**Proceed with caution.** Verify the domain carefully before sharing any data.")
            else:
                st.success("**This URL appears safe** based on available signals. Always double-check the domain name.")


# ═══════════════════════════════════════════════════════════════
#  SECTION I2 — BULK SCAN PAGE  (NEW FEATURE)
# ═══════════════════════════════════════════════════════════════

elif page == "Bulk Scan":

    st.markdown("## 📋 Bulk URL Scanner")
    st.markdown("<p style='color:#64748b; margin-top:-8px;'>Scan multiple URLs at once — paste them, upload a PDF containing links.</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models…"):
        try:
            m = train_models(st.session_state.dataset_bytes)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()

    # Input methods via tabs
    tab_paste, tab_pdf = st.tabs(["📝 Paste URLs", "📕 Upload PDF"])

    urls_to_scan = []

    with tab_paste:
        st.markdown('<div class="box"><div class="box-title">PASTE URLS (ONE PER LINE)</div>', unsafe_allow_html=True)
        url_text = st.text_area(
            "Paste URLs here, one per line:",
            height=200,
            placeholder="https://google.com\nhttps://suspicious-login-verify.com\nhttp://192.168.1.1/phish\nhttps://github.com",
            label_visibility="collapsed",
            key="bulk_paste",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        if url_text.strip():
            lines = [line.strip() for line in url_text.strip().split("\n") if line.strip()]
            urls_to_scan = lines

    with tab_pdf:
        st.markdown('<div class="box"><div class="box-title">UPLOAD A PDF WITH EMBEDDED LINKS</div>', unsafe_allow_html=True)
        pdf_file = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            label_visibility="collapsed",
            key="bulk_pdf",
        )
        st.markdown("""
        <small style='color:#475569'>
        The system will extract all hyperlinks and URLs found in the PDF text.
        For best results, install <code>PyMuPDF</code>: <code>pip install PyMuPDF</code>
        </small>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if pdf_file:
            pdf_bytes = pdf_file.read()
            extracted = extract_urls_from_pdf(pdf_bytes)
            if extracted:
                urls_to_scan = extracted
                st.info(f"Extracted **{len(extracted)}** unique URLs from the PDF.")
            else:
                st.warning("No URLs were found in this PDF. The file may not contain hyperlinks or recognizable URLs.")

    # Scan button
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if urls_to_scan:
        st.markdown(f"**{len(urls_to_scan)}** URLs ready to scan")

    scan_btn = st.button("🚀   SCAN ALL URLS", key="bulk_scan_btn")

    if scan_btn and urls_to_scan:
        results = []
        progress_bar = st.progress(0, text="Scanning URLs…")

        for i, raw_url in enumerate(urls_to_scan):
            url = raw_url.strip()
            if not url.startswith("http"):
                url = "http://" + url

            try:
                url_feats = extract_url_features(url)
                result = combined_predict(url, url_feats, m)
                results.append({
                    "url": url,
                    "verdict": result["verdict"],
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "rule_score": result["rule_score"],
                    "override": result.get("override"),
                })
            except Exception as exc:
                results.append({
                    "url": url,
                    "verdict": "error",
                    "label": f"ERROR: {exc}",
                    "confidence": 0,
                    "rule_score": 0,
                    "override": None,
                })

            progress_bar.progress((i + 1) / len(urls_to_scan), text=f"Scanning {i+1}/{len(urls_to_scan)}…")

        progress_bar.empty()

        # Summary cards
        n_safe = sum(1 for r in results if r["verdict"] == "safe")
        n_suspicious = sum(1 for r in results if r["verdict"] == "suspicious")
        n_phishing = sum(1 for r in results if r["verdict"] == "phishing")
        n_error = sum(1 for r in results if r["verdict"] == "error")

        st.markdown("### Scan Summary")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""
            <div class="summary-card">
              <div class="s-num" style="color:#22c55e;">{n_safe}</div>
              <div class="s-lbl">Legitimate</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="summary-card">
              <div class="s-num" style="color:#f59e0b;">{n_suspicious}</div>
              <div class="s-lbl">Suspicious</div>
            </div>
            """, unsafe_allow_html=True)
        with s3:
            st.markdown(f"""
            <div class="summary-card">
              <div class="s-num" style="color:#ef4444;">{n_phishing}</div>
              <div class="s-lbl">Phishing</div>
            </div>
            """, unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="summary-card">
              <div class="s-num" style="color:#64748b;">{len(results)}</div>
              <div class="s-lbl">Total Scanned</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Detailed results
        st.markdown("### Detailed Results")

        for r in results:
            v = r["verdict"]
            if v == "phishing":
                row_cls, tag_cls, tag_text = "bulk-phish", "tag-phish", "PHISHING"
            elif v == "suspicious":
                row_cls, tag_cls, tag_text = "bulk-warn", "tag-warn", "SUSPICIOUS"
            elif v == "safe":
                row_cls, tag_cls, tag_text = "bulk-safe", "tag-safe", "LEGITIMATE"
            else:
                row_cls, tag_cls, tag_text = "bulk-warn", "tag-warn", "ERROR"

            score_pct = int(r["confidence"] * 100)
            override_note = ""
            if r["override"] == "trusted_domain":
                override_note = " · 🛡️ Trusted"

            st.markdown(f"""
            <div class="bulk-row {row_cls}">
              <span class="bulk-tag {tag_cls}">{tag_text}</span>
              <span class="bulk-url">{r['url']}</span>
              <span class="bulk-score">{score_pct}%{override_note}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Download results as CSV
        st.markdown("### Export Results")
        export_df = pd.DataFrame([
            {
                "URL": r["url"],
                "Verdict": r["verdict"].upper(),
                "Label": r["label"],
                "Risk Score (%)": int(r["confidence"] * 100),
                "Rule Score": r["rule_score"],
            }
            for r in results
        ])
        st.dataframe(export_df, use_container_width=True, hide_index=True)

        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label="📥  Download Results as CSV",
            data=csv_data,
            file_name="bulk_scan_results.csv",
            mime="text/csv",
        )

    elif scan_btn and not urls_to_scan:
        st.warning("No URLs to scan. Please paste URLs, upload a .txt file, or upload a PDF first.")


# ═══════════════════════════════════════════════════════════════
#  SECTION J — DATASET ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════════

elif page == "Dataset":

    st.markdown("## 📊 Dataset & Model Training")
    st.markdown("<p style='color:#64748b; margin-top:-8px;'>Upload your own CSV dataset, explore it, and retrain the ensemble model.</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Upload widget
    st.markdown('<div class="box"><div class="box-title">📂 DATASET UPLOAD</div>', unsafe_allow_html=True)
    c_up1, c_up2 = st.columns([2, 1])
    with c_up1:
        uploaded = st.file_uploader(
            "Upload a custom CSV dataset",
            type=["csv"],
            help="CSV must have 30 feature columns and a Result/label column. "
                 "Labels should be -1 (phishing) and 1 (legitimate), or similar.",
            label_visibility="collapsed",
        )
        if uploaded:
            bytes_val = uploaded.read()
            if bytes_val != st.session_state.dataset_bytes:
                st.session_state.dataset_bytes = bytes_val
            st.success(f"Loaded: **{uploaded.name}**")
        else:
            st.info("No file uploaded — using the default **dataset.csv** (11,055 URLs · 30 features).")
    with c_up2:
        st.markdown("""
        <div style='padding:14px; background:rgba(56,189,248,0.04); border:1px solid rgba(56,189,248,0.1); border-radius:10px; font-size:0.82rem; color:#64748b; line-height:1.7;'>
        <b style='color:#38bdf8;'>Expected Format</b><br>
        • 30 feature columns<br>
        • A <code>Result</code> / <code>label</code> column<br>
        • Values: -1 (phishing), 1 (legitimate)<br>
        • CSV, UTF-8 encoded
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Train button
    if st.button("⚙️   RETRAIN MODEL"):
        train_models.clear()

    # Load/train models
    with st.spinner("Training ensemble (RF + SVM + MLP)…"):
        try:
            m = train_models(st.session_state.dataset_bytes)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    st.success(f"Models trained on **{m['source']}** — {m['n_samples']:,} samples · {m['n_features']} features")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Dataset preview
    st.markdown("### Dataset Preview")
    df_preview = m["df"]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Rows",     f"{df_preview.shape[0]:,}")
    r2.metric("Columns",  f"{df_preview.shape[1]}")
    r3.metric("Label col", m["label_col"])
    phish_count = (df_preview[m["label_col"]] == -1).sum()
    legit_count = len(df_preview) - phish_count
    r4.metric("Balance", f"{phish_count:,} / {legit_count:,}")

    st.dataframe(df_preview.head(10), use_container_width=True, hide_index=True)
    st.caption(f"Showing first 10 of {df_preview.shape[0]:,} rows.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Model accuracy section
    st.markdown("### Model Accuracy")
    acc_data = {
        "Random Forest": m["rf_acc"],
        "SVM (RBF)":     m["svm_acc"],
        "Neural Net":    m["mlp_acc"],
    }

    ac1, ac2, ac3 = st.columns(3)
    cols_acc = [ac1, ac2, ac3]
    colors   = ["#38bdf8", "#22c55e", "#f59e0b"]
    icons    = ["🌲", "⚡", "🧠"]
    for col, (mname, acc), color, ico in zip(cols_acc, acc_data.items(), colors, icons):
        pct = int(acc * 100)
        with col:
            st.markdown(f"""
            <div class="box" style="text-align:center;">
              <div class="box-title" style="justify-content:center;">{ico} {mname}</div>
              <div style="font-family:'JetBrains Mono',monospace; font-size:2rem;
                          color:{color};">
                {pct}%
              </div>
              <div class="acc-track" style="margin-top:10px;">
                <div class="acc-fill" style="width:{pct}%; background:{color};">
                  {acc*100:.2f}%
                </div>
              </div>
              <div style="font-size:0.76rem; color:#475569; margin-top:6px;">
                Test-set accuracy
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Visual accuracy bar chart
    st.markdown("#### Accuracy Comparison")
    acc_df = pd.DataFrame({
        "Model":    list(acc_data.keys()),
        "Accuracy": [v * 100 for v in acc_data.values()],
    })
    st.bar_chart(acc_df.set_index("Model"), color="#38bdf8", height=280)


# ═══════════════════════════════════════════════════════════════
#  SECTION K — ABOUT PAGE
# ═══════════════════════════════════════════════════════════════

elif page == "About":

    st.markdown("## ℹ️ About EnsemblePhish")
    st.markdown("<p style='color:#64748b; margin-top:-8px;'>Learn about phishing, ensemble ML, and how this system works.</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # What is phishing
    st.markdown("### What is Phishing?")
    st.markdown("""
    <div class="step-card">
      <h5>DEFINITION</h5>
      <p>
        Phishing is a cyber-attack where criminals create fake websites that mimic trusted brands
        (banks, email providers, e-commerce) to steal usernames, passwords, credit card numbers,
        or other sensitive data. The victim is lured via an email, SMS, or social media link.
      </p>
    </div>
    <div class="step-card">
      <h5>COMMON TACTICS</h5>
      <p>
        Raw IP addresses instead of domain names · URL shorteners to hide destinations ·
        Fake HTTPS or missing SSL certificates · Hyphenated domains like <code>paypal-secure.com</code> ·
        Excessive subdomains like <code>login.verify.paypal.phish.com</code> ·
        Suspicious keywords (verify, secure, update, confirm) · @ symbols to trick browsers.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Ensemble Learning
    st.markdown("### What is Ensemble Learning?")
    st.markdown("""
    <div class="step-card">
      <h5>CORE IDEA</h5>
      <p>
        Instead of relying on a single ML model (which may have blind spots), ensemble learning
        combines <b>multiple models</b> and takes a majority vote. If 2 out of 3 models say
        "phishing", the ensemble verdict is phishing. This reduces variance and improves
        generalization far beyond any individual model.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Models used
    st.markdown("### Models Used")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="model-card">
          <div class="model-icon">🌲</div>
          <h5>RANDOM FOREST</h5>
          <p>
            An ensemble of 100 decision trees, each trained on a random subset of features
            and samples. Robust to outliers, fast to train, and provides feature importance
            rankings. Best all-rounder for tabular data.
          </p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="model-card">
          <div class="model-icon">⚡</div>
          <h5>SVM (RBF KERNEL)</h5>
          <p>
            Support Vector Machine with a Radial Basis Function kernel. Projects data into
            a high-dimensional space to find an optimal separating hyperplane. Highly effective
            when classes are not linearly separable.
          </p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="model-card">
          <div class="model-icon">🧠</div>
          <h5>MLP NEURAL NETWORK</h5>
          <p>
            A Multi-Layer Perceptron with two hidden layers (64 → 32 neurons). Learns
            non-linear feature combinations that tree-based and kernel methods can miss.
            Trained for 300 epochs with adam optimiser.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # How the system works
    st.markdown("### How the System Works")
    for title, body in [
        ("STEP 1 · WHITELIST CHECK",
         "The base domain is extracted and compared against a list of 30+ globally trusted domains. If matched, the verdict is instantly SAFE without consulting ML."),
        ("STEP 2 · FEATURE EXTRACTION",
         "9 features are extracted from the URL string: IP detection, length, shortener, @ symbol, double slash, hyphen, subdomain count, SSL state, and HTTPS token."),
        ("STEP 3 · RULE-BASED SCORING",
         "Each extracted feature maps to a risk score (0–100). Strong signals (raw IP, @, no HTTPS) can override the ML result entirely if score ≥ 70."),
        ("STEP 4 · ML ENSEMBLE PREDICTION",
         "The 9 real features + 21 legitimate-class medians form a 30-d vector. All 3 models output phishing probabilities. Majority vote decides the ML verdict."),
        ("STEP 5 · BLENDED FINAL DECISION",
         "Final score = 60% × ML_prob + 40% × rule_score/100. Thresholds: <40% = Safe, 40–64% = Suspicious, ≥65% = Phishing."),
    ]:
        st.markdown(f"""
        <div class="step-card">
          <h5>{title}</h5>
          <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Dataset info
    st.markdown("### Dataset")
    st.markdown("""
    <div class="step-card">
      <h5>IEEE ICTBIG-2023 PHISHING DATASET</h5>
      <p>
        11,055 URLs · 30 engineered features · Binary labels (-1 phishing / 1 legitimate) ·
        Published at the IEEE ICTBIG-2023 conference. Features span URL structure, domain metadata,
        SSL state, page content signals, and external reputation metrics.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
  EnsemblePhish &nbsp;·&nbsp; Python + Streamlit + scikit-learn
  &nbsp;·&nbsp; IEEE ICTBIG-2023 Dataset &nbsp;·&nbsp; Ensemble ML + Rule Engine
</div>
""", unsafe_allow_html=True)