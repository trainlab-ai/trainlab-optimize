"""Synthetic but realistic support data for Northwind Cloud, a fictional file-sync SaaS.

Everything is generated deterministically from a seed, so two runs send the same tickets.
Tickets contain names, emails and phone numbers on purpose: captured samples are redacted.
"""
from __future__ import annotations

import random

QUEUES = ["billing", "bug", "account", "shipping", "feature_request"]

PRODUCTS = ["Northwind Drive", "Northwind Sync for Teams", "Northwind Backup", "Northwind Mobile", "Northwind Desk Hub"]
SOFTWARE = PRODUCTS[:4]          # the Desk Hub is hardware: it appears only in order and shipping tickets
PLANS = ["Starter", "Pro", "Business", "Enterprise"]
NAMES = ["Priya Nair", "Tom Becker", "Aisha Khan", "Luis Ortega", "Mei Lin", "Sam O'Neill", "Fatima Zahra",
         "Jonas Weber", "Ana Souza", "Kenji Sato", "Olivia Brown", "Ravi Iyer", "Chloe Martin", "Diego Ruiz"]
CITIES = ["Pune", "Leeds", "Austin", "Lyon", "Osaka", "Toronto", "Porto", "Munich", "Nairobi", "Melbourne"]

TEMPLATES = {
    "billing": [
        ("Charged twice this month", "Hi, I was billed twice for my {plan} plan on the 3rd. Both charges show on my card "
         "ending 4417. Please refund the duplicate. Account email {email}."),
        ("Invoice shows wrong VAT number", "Our invoice for {month} lists the old VAT number. We moved to {city} last "
         "quarter. Can you reissue it with the new one? Finance needs it by Friday."),
        ("Downgrade did not reduce price", "I downgraded from {plan} to Starter two weeks ago but was still charged the "
         "full {plan} price. What happened?"),
        ("Need a refund for unused seats", "We removed 12 seats from {product} on the 10th. Are we getting a prorated "
         "credit for those or a refund? Our contract says refund."),
        ("Card declined but I have funds", "Payment for {plan} keeps failing with 'card declined'. The bank says nothing "
         "is blocked. Can you retry or give me another way to pay? Call me on {phone}."),
    ],
    "bug": [
        ("Sync stuck at 99%", "{product} has been stuck at 99% syncing for two days. Restarting doesn't help. "
         "Windows 11, version 8.4.2. About 40,000 files in the folder."),
        ("App crashes when opening shared folder", "Every time I open the shared folder 'Q3 Board' in {product} the "
         "app closes without an error. Other folders open fine."),
        ("Conflict copies everywhere", "Since the last update {product} creates 'conflicted copy' files for documents "
         "nobody else edited. We now have hundreds of duplicates."),
        ("Search returns no results", "Search in {product} returns nothing for files I can see in the folder. It "
         "worked last week. Re-indexing didn't fix it."),
        ("Upload fails over 2 GB", "Uploading a 2.6 GB video to {product} fails at around 80% with error NW-E5021. "
         "Smaller files upload fine."),
    ],
    "account": [
        ("Can't log in after password reset", "I reset my password but the new one is rejected. I've tried three "
         "times. My login is {email}. Please help, I'm locked out."),
        ("Transfer ownership to a colleague", "I'm leaving the company. How do I transfer ownership of our {plan} "
         "workspace to {name2}?"),
        ("Two-factor phone lost", "I lost the phone with my authenticator app. I can't get past 2FA. My backup "
         "number is {phone}."),
        ("Merge two accounts", "I accidentally created a second account with my personal email. Can you merge it "
         "into my work account {email}?"),
        ("SSO login loop", "After enabling SSO for our domain, users get redirected back to the login page forever. "
         "We're on {plan}."),
    ],
    "shipping": [
        ("Where is my Desk Hub?", "I ordered a Northwind Desk Hub, order {order}, ten days ago and tracking hasn't "
         "updated since it left the warehouse. Can you check?"),
        ("Wrong item delivered", "Order {order} arrived but it's the 2-port hub, not the 4-port one I paid for. "
         "How do I swap it?"),
        ("Change delivery address", "Can I change the delivery address for order {order}? I've moved to {city}."),
        ("Package marked delivered but not here", "Tracking for order {order} says delivered yesterday but nothing "
         "arrived. Neighbours haven't seen it either."),
        ("Customs fee on my order", "I was asked to pay a customs fee for order {order}. The checkout said duties "
         "were included. Who covers this?"),
    ],
    "feature_request": [
        ("Please add dark mode", "Any plans for dark mode in {product}? Our team works late and the white UI is "
         "harsh."),
        ("Selective sync for Linux", "We'd love selective sync on the Linux client of {product}, like on Windows and "
         "Mac. It's the one thing blocking our rollout."),
        ("Export audit log to CSV", "As a {plan} admin I need to export the audit log to CSV for our quarterly review. "
         "Is this on the roadmap?"),
        ("Shared links with expiry", "It would help if shared links in {product} could expire after a set number of "
         "days. Our legal team keeps asking."),
        ("API for usage reports", "Could you expose seat and storage usage through the API? We want to feed it into "
         "our own dashboards."),
    ],
}

KB_ARTICLES = [
    ("Refunds and duplicate charges", "Duplicate charges are refunded automatically within 5 business days once "
     "support confirms them. Prorated credits for removed seats are applied to the next invoice; contract customers "
     "on annual terms can request a refund to the original payment method."),
    ("Updating invoice details", "Admins can change the billing address and VAT number under Settings > Billing. "
     "Past invoices can be reissued once with the new details from the same page."),
    ("Fixing a stuck sync", "If sync stops near 100%, pause and resume sync, then run Repair from the tray menu. "
     "Folders with more than 30,000 files sync faster with 'Large folder mode' turned on."),
    ("Conflict copies", "Conflicted copies appear when two devices edit offline. Version 8.4.2 had a bug that created "
     "false conflicts; update to 8.4.3 and run 'Clean conflicts' to merge duplicates safely."),
    ("Large uploads", "Uploads over 2 GB need the desktop app 8.4 or later, or the web uploader with resumable mode. "
     "Error NW-E5021 means the browser upload timed out."),
    ("Account access and 2FA", "Locked-out users can verify with a backup code or a backup phone number. Support can "
     "reset 2FA after identity verification by the workspace admin."),
    ("Workspace ownership", "Owners transfer a workspace under Settings > Members > Transfer ownership. The new owner "
     "must already be an admin."),
    ("Single sign-on", "An SSO login loop usually means the ACS URL in the identity provider doesn't match the one in "
     "Settings > Security. Copy it again and clear the browser cookies."),
    ("Hardware orders and returns", "Desk Hub orders ship in 2 business days. Wrong items are exchanged free with a "
     "prepaid label. Duties are included for EU, UK and US deliveries; elsewhere they are refunded on request."),
    ("Roadmap and feature requests", "Feature requests are reviewed monthly. Dark mode and Linux selective sync are "
     "planned for the next two releases; audit log export is available on Business and Enterprise today."),
]

ORDERS = {}   # filled by tickets(): order id -> status record


def _email(name: str, rng: random.Random) -> str:
    first, last = name.lower().replace("'", "").split()[:2]
    return f"{first}.{last}@{rng.choice(['acme-retail.com', 'globex.io', 'initech.co', 'umbrella.org'])}"


def _phone(rng: random.Random) -> str:
    return f"+44 7{rng.randint(100, 999)} {rng.randint(100000, 999999)}"


def tickets(n: int = 60, seed: int = 7) -> list[dict]:
    """n tickets, spread across the five queues, each with its true queue in 'label'."""
    rng = random.Random(seed)
    out = []
    statuses = ["in transit", "delivered", "held at customs", "processing", "out for delivery"]
    for i in range(n):
        queue = QUEUES[i % len(QUEUES)] if i < len(QUEUES) * 2 else rng.choice(QUEUES)
        subject, body = rng.choice(TEMPLATES[queue])
        name, name2 = rng.sample(NAMES, 2)
        order = f"NW-{rng.randint(10000, 99999)}"
        fields = {"plan": rng.choice(PLANS), "product": rng.choice(SOFTWARE), "email": _email(name, rng),
                  "phone": _phone(rng), "city": rng.choice(CITIES), "month": rng.choice(["July", "August", "September"]),
                  "order": order, "name2": name2}
        if queue == "shipping":
            ORDERS[order] = {"order_id": order, "status": rng.choice(statuses), "item": "Desk Hub 4-port",
                             "carrier": rng.choice(["DHL", "UPS", "DPD"]), "shipped_days_ago": rng.randint(1, 12)}
        out.append({"id": f"T-{1000 + i}", "customer": name, "subject": subject,
                    "body": body.format(**fields) + f"\n\n{name}", "label": queue})
    return out
