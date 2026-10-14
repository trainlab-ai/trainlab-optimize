"""Cloud billing exports: AWS Cost and Usage Report, Azure Cost Management export, GCP billing export.

Spend always comes from the export's cost column (authoritative). The model name and whether
a line is input or output tokens are parsed from the usage type / meter / SKU description,
which differ between clouds and over time, so treat model names from these sources as
best-effort and check the "Spend by model" table against your bill.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from ..schema import UsageRecord

AWS_COLS = {
    "product": ["lineItem/ProductCode", "line_item_product_code"],
    "usage_type": ["lineItem/UsageType", "line_item_usage_type"],
    "date": ["lineItem/UsageStartDate", "line_item_usage_start_date"],
    "cost": ["lineItem/UnblendedCost", "line_item_unblended_cost", "lineItem/NetUnblendedCost"],
    "amount": ["lineItem/UsageAmount", "line_item_usage_amount"],
    "unit": ["pricing/unit", "pricing_unit"],
    "account": ["lineItem/UsageAccountId", "line_item_usage_account_id"],
    "description": ["lineItem/LineItemDescription", "line_item_line_item_description"],
}
AZURE_COLS = {
    "date": ["Date", "date", "UsageDate", "usageDate"],
    "category": ["MeterCategory", "meterCategory"],
    "subcategory": ["MeterSubCategory", "meterSubCategory", "MeterSubcategory"],
    "meter": ["MeterName", "meterName"],
    "quantity": ["Quantity", "quantity", "UsageQuantity"],
    "unit": ["UnitOfMeasure", "unitOfMeasure"],
    "cost": ["CostInUsd", "costInUsd", "CostInBillingCurrency", "costInBillingCurrency", "Cost", "PreTaxCost"],
    "resource": ["ResourceName", "resourceName", "ResourceGroup", "resourceGroupName", "ResourceId"],
}
GCP_COLS = {
    "service": ["service.description", "service_description", "Service description", "Service"],
    "sku": ["sku.description", "sku_description", "SKU description", "SKU"],
    "date": ["usage_start_time", "Usage start date", "usage_start_date"],
    "cost": ["cost", "Cost ($)", "Cost"],
    "amount": ["usage.amount", "usage_amount", "Usage amount"],
    "unit": ["usage.unit", "usage_unit", "Usage unit"],
    "project": ["project.id", "project_id", "Project ID"],
}


def _pick(header: list[str], spec: dict) -> dict:
    lower = {h.lower(): h for h in header}
    out = {}
    for field, names in spec.items():
        for n in names:
            if n in header:
                out[field] = n
                break
            if n.lower() in lower:
                out[field] = lower[n.lower()]
                break
    return out


def detect(header: list[str]) -> str | None:
    if {"product", "usage_type", "cost"} <= _pick(header, AWS_COLS).keys():
        return "aws"
    if {"meter", "cost"} <= _pick(header, AZURE_COLS).keys():
        return "azure"
    if {"service", "sku", "cost"} <= _pick(header, GCP_COLS).keys():
        return "gcp"
    return None


def _num(v) -> float:
    try:
        return float(str(v).replace(",", "").replace("$", "").strip() or 0)
    except ValueError:
        return 0.0


def _direction(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(output|outp|completion|generated)\b|-output|output-tokens|outp-", t):
        return "out"
    if re.search(r"\b(input|inp|prompt|context)\b|-input|input-tokens|inp-|cache", t):
        return "in"
    return None


def _tokens(amount: float, unit: str, text: str) -> int:
    u = f"{unit} {text}".lower()
    if "token" not in u and "characters" not in u:
        return 0
    mult = 1_000_000 if re.search(r"\b1m\b|million", u) else 1000 if re.search(r"\b1k\b|thousand|1,000", u) else 1
    return int(amount * mult)


# ---- model name extraction --------------------------------------------------------------

def _aws_model(usage_type: str) -> str:
    # USE1-Claude3.5Sonnet-input-tokens / USW2-MP:USW2_InputTokenCount-Units / EU-Llama3-70B-output-tokens
    m = re.sub(r"^[A-Z]{2,4}\d?-", "", usage_type)
    m = re.sub(r"[-_ ](input|output)[-_ ]?tokens?.*$", "", m, flags=re.I)
    m = re.sub(r"[-_ ](cache[-_ ]?(read|write)).*$", "", m, flags=re.I)
    return m or usage_type


def _azure_model(meter: str) -> str:
    # "gpt-4o-0806-Inp-glbl 1K Tokens", "gpt 4o mini Outp regional 1K Tokens", "text-embedding-3-small 1K Tokens"
    m = re.split(r"[-\s](?:Inp|Outp|Input|Output|cchd|Cached|Batch|Btch)\b", meter, maxsplit=1, flags=re.I)[0]
    m = re.sub(r"\s*\d+[KM]\s*Tokens?.*$", "", m, flags=re.I).strip()
    return re.sub(r"\s+", "-", m).lower()


def _gcp_model(sku: str) -> str:
    # "Gemini 2.5 Flash Input Text", "Generate content input token count Gemini 2.5 Pro",
    # "Claude Sonnet 4.5 Output Tokens"
    m = re.search(r"(gemini[\s-][\d.]+(?:[\s-](?:flash|pro|ultra|lite|nano))*(?:[\s-]lite)?|"
                  r"claude[\s-][a-z]+[\s-][\d.]+|claude[\s-][\d.]+[\s-][a-z]+|"
                  r"text-embedding[\w.-]*|gemma[\s-][\w.]+|llama[\s-][\w.]+)", sku, re.I)
    name = m.group(1) if m else sku
    name = re.sub(r"\s+", "-", name.strip()).lower()
    return name.replace("claude-sonnet-", "claude-sonnet-").replace(".", "-") if "claude" in name else name


def read_cloud_csv(path: str | Path, kind: str | None = None) -> list[UsageRecord]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        kind = kind or detect(header)
        if kind is None:
            raise ValueError(f"{path}: not a recognised AWS/Azure/GCP billing export")
        rows = list(reader)

    out: list[UsageRecord] = []
    src = f"{kind}:{path.name}"
    if kind == "aws":
        c = _pick(header, AWS_COLS)
        for r in rows:
            product = (r.get(c["product"]) or "")
            if "bedrock" not in product.lower():
                continue
            ut = r.get(c["usage_type"]) or ""
            desc = r.get(c.get("description", ""), "") or ""
            cost = _num(r.get(c["cost"]))
            if not cost and not _num(r.get(c.get("amount", ""), 0)):
                continue
            direction = _direction(ut + " " + desc)
            tokens = _tokens(_num(r.get(c.get("amount", ""), 0)), r.get(c.get("unit", ""), "") or "", ut)
            out.append(UsageRecord(
                date=(r.get(c.get("date", ""), "") or "")[:10] or "unknown", provider="bedrock",
                model=_aws_model(ut), input_tokens=tokens if direction == "in" else 0,
                output_tokens=tokens if direction == "out" else 0, cost_usd=cost,
                project=r.get(c.get("account", ""), None), source=src))
    elif kind == "azure":
        c = _pick(header, AZURE_COLS)
        for r in rows:
            cat = f"{r.get(c.get('category', ''), '')} {r.get(c.get('subcategory', ''), '')}".lower()
            meter = r.get(c["meter"]) or ""
            if not ("openai" in cat or "foundry" in cat or "cognitive" in cat or "ai services" in cat
                    or re.search(r"gpt|o\d|embedding|claude|llama|phi|mistral", meter, re.I)):
                continue
            if not re.search(r"token", meter + " " + (r.get(c.get("unit", ""), "") or ""), re.I):
                continue
            direction = _direction(meter)
            tokens = _tokens(_num(r.get(c.get("quantity", ""), 0)), r.get(c.get("unit", ""), "") or "", meter)
            out.append(UsageRecord(
                date=_azure_date(r.get(c.get("date", ""), "")), provider="azure",
                model="azure/" + _azure_model(meter),
                input_tokens=tokens if direction != "out" else 0,
                output_tokens=tokens if direction == "out" else 0,
                cached_input_tokens=tokens if re.search(r"cchd|cached", meter, re.I) else 0,
                cost_usd=_num(r.get(c["cost"])),
                project=(r.get(c.get("resource", ""), "") or None), source=src))
    else:
        c = _pick(header, GCP_COLS)
        for r in rows:
            service = (r.get(c["service"]) or "").lower()
            if not any(k in service for k in ("vertex", "gemini", "generative language", "generative ai")):
                continue
            sku = r.get(c["sku"]) or ""
            direction = _direction(sku)
            tokens = _tokens(_num(r.get(c.get("amount", ""), 0)), r.get(c.get("unit", ""), "") or "", sku)
            out.append(UsageRecord(
                date=(r.get(c.get("date", ""), "") or "")[:10] or "unknown", provider="gemini",
                model=_gcp_model(sku), input_tokens=tokens if direction != "out" else 0,
                output_tokens=tokens if direction == "out" else 0, cost_usd=_num(r.get(c["cost"])),
                project=r.get(c.get("project", ""), None), source=src))
    return out


def _azure_date(v: str) -> str:
    v = (v or "").strip()
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", v)          # MM/DD/YYYY
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return v[:10] or "unknown"
