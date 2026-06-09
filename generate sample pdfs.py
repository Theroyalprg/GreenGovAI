"""
generate_sample_pdfs.py
Generates realistic placeholder PDF documents for each scheme.
Run this ONCE to populate /data if you don't have the real PDFs.
Install: pip install fpdf2
"""

from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("Install fpdf2 first:  pip install fpdf2")
    raise

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SCHEMES = {
    "PM-KUSUM": {
        "full_name": "Pradhan Mantri Kisan Urja Suraksha evam Utthaan Mahabhiyan (PM-KUSUM)",
        "ministry": "Ministry of New and Renewable Energy",
        "objective": "To provide energy security to farmers along with honouring India's commitment "
                     "to increase the share of installed capacity of electric power from non-fossil "
                     "fuel-based energy resources to 40% by 2030.",
        "components": [
            "Component A: 10,000 MW of Decentralized Ground/Stilt Mounted Grid Connected Solar or "
            "other Renewable Energy Based Power Plants",
            "Component B: Installation of 20 lakh standalone Solar Powered Agriculture Pumps",
            "Component C: Solarisation of 15 lakh Grid-connected Solar Powered Agriculture Pumps",
        ],
        "eligibility": [
            "Individual farmers",
            "Group of farmers / Farmer Producer Organisations (FPO)",
            "Water User Associations (WUA)",
            "Panchayats",
            "Co-operatives",
        ],
        "subsidy": "Central Financial Assistance (CFA) of 30% of the benchmark cost. "
                   "State governments provide an additional 30% subsidy. "
                   "Farmers need to bear only 40% of the cost (with bank loan facility available).",
        "documents": [
            "Aadhaar card",
            "Land ownership documents / Khasra-Khatauni",
            "Bank account details",
            "Electricity connection details (for Component C)",
            "Passport-size photographs",
        ],
        "application": [
            "Visit the official MNRE portal: https://mnre.gov.in",
            "Register on your State Nodal Agency (SNA) website",
            "Fill the online application form with land and bank details",
            "Upload required documents",
            "Pay the nominal application fee (state-specific)",
            "Await approval and disbursement notification via SMS/email",
        ],
    },
    "PMKSY": {
        "full_name": "Pradhan Mantri Krishi Sinchayee Yojana (PMKSY)",
        "ministry": "Ministry of Jal Shakti / Ministry of Agriculture & Farmers Welfare",
        "objective": "Achieve convergence of investments in irrigation at the field level, "
                     "expand cultivable area under assured irrigation, improve on-farm water "
                     "use efficiency, introduce sustainable water conservation practices.",
        "components": [
            "Har Khet Ko Pani: Expansion of cultivable area under irrigation",
            "More Crop Per Drop: Micro-irrigation (drip/sprinkler)",
            "Watershed Development: Groundwater recharge and runoff management",
            "AIBP: Accelerated Irrigation Benefits Programme for incomplete projects",
        ],
        "eligibility": [
            "All farmers with agricultural land",
            "Small and marginal farmers get priority (up to 2 hectares)",
            "Farmers in water-stressed districts get additional benefits",
            "Self Help Groups (SHGs) and cooperatives",
        ],
        "subsidy": "55% subsidy for small and marginal farmers; 45% for other farmers. "
                   "Additional top-up by state governments in many states.",
        "documents": [
            "Aadhaar card and PAN",
            "Land record (Khasra/Patta)",
            "Bank passbook copy",
            "Caste certificate (if SC/ST)",
            "Water source availability certificate",
        ],
        "application": [
            "Visit the PMKSY portal: https://pmksy.gov.in",
            "Contact District Agriculture Office",
            "Submit application with land documents",
            "Field verification by agriculture officer",
            "Installation by empanelled vendor",
            "Subsidy disbursement to vendor directly",
        ],
    },
    "Soil_Health_Card": {
        "full_name": "Soil Health Card Scheme",
        "ministry": "Department of Agriculture & Farmers Welfare",
        "objective": "Issue Soil Health Cards to farmers which will carry crop-wise "
                     "recommendations of nutrients and fertilisers required for individual farms "
                     "to help farmers improve productivity through judicious use of inputs.",
        "components": [
            "Free soil sample collection from farmer fields",
            "Laboratory testing for 12 parameters",
            "Printing and distribution of Soil Health Cards",
            "Training of farmers on card interpretation",
        ],
        "eligibility": [
            "All farmers across India — no income or land size restriction",
            "Applicable to all cropping seasons",
            "New cards issued every 2 years",
        ],
        "subsidy": "Completely free of cost — soil testing, card printing, and delivery "
                   "are funded by the Government of India.",
        "documents": [
            "Aadhaar card",
            "Land record (optional for initial registration)",
        ],
        "application": [
            "Contact nearest Krishi Vigyan Kendra (KVK) or Agriculture Department office",
            "Register online at: https://soilhealth.dac.gov.in",
            "Soil sample collected by government officials",
            "Card delivered within 30 days",
            "Get free consultation from agriculture officer",
        ],
    },
    "PMFBY": {
        "full_name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "objective": "Provide financial support to farmers suffering crop loss/damage due to "
                     "unforeseen events. Stabilise the income of farmers and encourage "
                     "modern agricultural practices.",
        "components": [
            "Kharif crops: 2% premium by farmer",
            "Rabi crops: 1.5% premium by farmer",
            "Annual commercial and horticultural crops: 5% by farmer",
            "Balance premium shared equally by Centre and State",
        ],
        "eligibility": [
            "All farmers growing notified crops in notified areas",
            "Compulsory for farmers with crop loans (loanee farmers)",
            "Voluntary for non-loanee farmers",
            "Sharecroppers and tenant farmers are also eligible",
        ],
        "subsidy": "Government of India pays 50% of the remaining premium after farmer's share. "
                   "States contribute the other 50%. Technology-based claim settlement within 45 days.",
        "documents": [
            "Aadhaar card",
            "Bank account details",
            "Land record / crop sowing certificate",
            "Loan documents (for loanee farmers)",
        ],
        "application": [
            "Visit pmfby.gov.in or Crop Insurance Portal",
            "Apply through nearest bank or Common Service Centre (CSC)",
            "Submit before the cut-off date of each season",
            "Receive insurance policy number via SMS",
            "Report crop loss to insurer within 72 hours of occurrence",
            "Claim processed after crop-cutting experiments",
        ],
    },
    "Micro_Irrigation": {
        "full_name": "National Mission on Micro Irrigation / PMKSY More Crop Per Drop",
        "ministry": "Ministry of Agriculture and Farmers Welfare",
        "objective": "Promote efficient water use through drip and sprinkler irrigation systems. "
                     "Reduce water consumption by 40-70% and increase crop productivity.",
        "components": [
            "Drip Irrigation Systems (suitable for orchards, vegetables, cotton, sugarcane)",
            "Sprinkler Irrigation Systems (suitable for wheat, groundnut, vegetables)",
            "Micro Sprinkler Systems for horticulture",
            "Training and capacity building",
        ],
        "eligibility": [
            "All categories of farmers",
            "Small and marginal farmers (up to 2 ha): 55% subsidy",
            "Other farmers (above 2 ha): 45% subsidy",
            "Cluster-based applications by SHGs or FPOs receive priority",
        ],
        "subsidy": "55% subsidy for small/marginal farmers. 45% for others. "
                   "Additional 10-15% from states like Maharashtra, Gujarat, Andhra Pradesh.",
        "documents": [
            "Aadhaar card",
            "Land record",
            "Bank account",
            "Quotation from empanelled supplier",
            "Irrigation source evidence (borewell/canal/tank)",
        ],
        "application": [
            "Apply at State Horticulture/Agriculture Department",
            "Online: https://pmksy.gov.in",
            "Choose empanelled vendor from approved list",
            "Get pre-installation inspection approved",
            "System installed by vendor",
            "Post-installation verification",
            "Subsidy credited directly to bank account",
        ],
    },
}


class SchemePDF(FPDF):
    def header(self):
        self.set_fill_color(13, 59, 30)
        self.rect(0, 0, 210, 22, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.cell(0, 14, "  Government of India — Scheme Information Document", ln=True, align="L")
        self.set_y(24)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"GreenGov AI Knowledge Base — Page {self.page_no()}", align="C")


def build_pdf(scheme_key: str, data: dict):
    pdf = SchemePDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=16)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(13, 59, 30)
    pdf.multi_cell(0, 9, data["full_name"], ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, f"Ministry: {data['ministry']}", ln=True)
    pdf.ln(4)

    def section(title):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(26, 107, 56)
        pdf.set_fill_color(232, 247, 238)
        pdf.cell(0, 8, f"  {title}", ln=True, fill=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)

    # Objective
    section("Objective")
    pdf.multi_cell(0, 6, data["objective"])
    pdf.ln(3)

    # Components
    section("Key Components / Pillars")
    for c in data["components"]:
        pdf.multi_cell(0, 6, f"  • {c}")
    pdf.ln(3)

    # Eligibility
    section("Eligibility Criteria")
    for e in data["eligibility"]:
        pdf.multi_cell(0, 6, f"  • {e}")
    pdf.ln(3)

    # Subsidy
    section("Subsidy / Financial Assistance")
    pdf.multi_cell(0, 6, data["subsidy"])
    pdf.ln(3)

    # Documents
    section("Required Documents")
    for d in data["documents"]:
        pdf.multi_cell(0, 6, f"  • {d}")
    pdf.ln(3)

    # Application Steps
    section("How to Apply — Step by Step")
    for i, step in enumerate(data["application"], 1):
        pdf.multi_cell(0, 6, f"  {i}. {step}")
    pdf.ln(3)

    # Footer notice
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(140, 100, 30)
    pdf.multi_cell(
        0, 6,
        "Note: Subsidy percentages and eligibility may vary by state. "
        "Always verify with official government portals before applying."
    )

    out = DATA_DIR / f"{scheme_key}.pdf"
    pdf.output(str(out))
    print(f"✅  {out}")


if __name__ == "__main__":
    print("Generating sample scheme PDFs in /data ...\n")
    for key, info in SCHEMES.items():
        build_pdf(key, info)
    print(f"\nDone — {len(SCHEMES)} PDFs created in {DATA_DIR.resolve()}")
