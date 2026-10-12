"""Generate the FICTIONAL sample PDFs used by tests and evaluation.

All names, numbers and policies below are invented. Run:  python data/generate_sample_documents.py
Requires reportlab (see requirements-dev.txt). The generated PDFs are committed, so end users
do not need to run this.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).parent / "sample_documents"

DOCUMENTS = {
    "Aurora_Labs_Employee_Handbook.pdf": [
        ("1. Welcome and Working Hours",
         "Welcome to Aurora Labs, a fictional software company headquartered in Marlow Quay. This handbook explains "
         "how we work together. The standard working week at Aurora Labs is 38 hours. Core hours are 10:00 to 15:00 "
         "local time, during which every team member should be available for meetings and collaboration. Outside "
         "core hours, employees may arrange their schedule flexibly with their manager. New employees begin with a "
         "probation period of 90 days, after which a performance review decides whether the contract continues on a "
         "permanent basis. During probation, either side may end the contract with one week of notice. Managers "
         "hold a one-to-one meeting with each new employee every week during the first month."),
        ("2. Leave Policy",
         "Full-time employees receive 27 days of annual leave per calendar year, plus 3 additional wellbeing days "
         "that can be taken at short notice. Up to 5 unused annual leave days may be carried over into the next "
         "year; any further days expire on 31 March. Primary caregivers receive 20 weeks of parental leave at full "
         "pay, while secondary caregivers receive 8 weeks at full pay. Sick leave is paid in full for up to 12 days "
         "per year. A doctor's note is required once an absence lasts more than 3 consecutive days. Employees "
         "should notify their manager before the start of the working day when they are unwell."),
        ("3. Remote Work and Expenses",
         "Employees may work remotely up to 3 days per week, subject to team needs. Aurora Labs pays a one-time home "
         "office stipend of 400 euros to help employees equip a workspace, and reimburses internet costs of 30 euros "
         "per month. Expense claims must be submitted within 45 days of the purchase date, otherwise they are not "
         "reimbursed. While travelling for business, meals are reimbursed up to 55 euros per day. Flights shorter "
         "than 6 hours must be booked in economy class. Receipts are required for every claim above 25 euros."),
        ("4. Information Security",
         "All employees must use the company password manager, called VaultKey, and enable multi-factor "
         "authentication on every work account. Any suspected security incident must be reported within 4 hours to "
         "the security team at security@auroralabs.example. Company laptops use full-disk encryption, and "
         "connecting personal USB drives is prohibited. Annual security awareness training takes place every March "
         "and is mandatory for all staff. Visitors must be accompanied at all times inside the office."),
    ],
    "Nimbus_X2_Drone_Manual.pdf": [
        ("1. Product Specifications",
         "The Nimbus X2 is a fictional consumer camera drone. It weighs 612 grams with its battery installed. "
         "Maximum flight time is 31 minutes under calm conditions. The maximum speed is 58 km/h and the maximum "
         "flight altitude is 120 metres above the take-off point. The transmission range is up to 6 km in open "
         "areas without interference. The built-in camera has a 20 megapixel sensor and records 4K video at 60 "
         "frames per second. The drone folds to fit inside the included carrying case."),
        ("2. Battery and Charging",
         "The Nimbus X2 uses the NB-5200 intelligent battery with a capacity of 5200 mAh. A full charge from empty "
         "takes 85 minutes with the supplied charger. Only charge the battery when the ambient temperature is "
         "between 5 and 40 degrees Celsius. For storage longer than ten days, keep the battery at about 60 percent "
         "charge in a cool, dry place. The battery should be replaced after 300 charge cycles. A blinking red LED on "
         "the battery indicates that the charge level is below 15 percent."),
        ("3. Flight Safety",
         "Do not fly within 5 km of an airport. Do not fly when wind speeds exceed 38 km/h, because the drone may "
         "be unable to hold its position. The Return-to-Home function activates automatically when the battery "
         "drops to 20 percent or when the control signal is lost for 8 seconds. Obstacle sensors are installed on "
         "the front and the bottom of the drone only, so it cannot detect obstacles to the sides or rear. The "
         "operating temperature range is minus 10 to 40 degrees Celsius. Always keep the drone within visual line "
         "of sight."),
        ("4. Maintenance and Troubleshooting",
         "Inspect the propellers for cracks before every flight, and replace them after 50 flight hours. Firmware "
         "updates are installed through the NimbusLink mobile app. If the gimbal appears tilted, run the gimbal "
         "calibration from the app settings menu. Error code E07 means compass interference; move away from metal "
         "structures and recalibrate the compass. Error code E12 means the gimbal motor is stalled; remove the "
         "gimbal guard and restart the drone. The Nimbus X2 comes with a warranty of 18 months from the purchase "
         "date."),
    ],
    "Greenfield_Library_Annual_Report_2025.pdf": [
        ("1. Membership and Visits",
         "Greenfield Public Library is a fictional municipal library. In 2025 the library had 18,420 active members, "
         "including 5,310 children's cards. Total visits reached 412,000, which is 9 percent higher than in 2024. "
         "The increase is mainly attributed to the introduction of Saturday opening hours and to new evening "
         "programmes for families. The library employed 41 staff members at the end of the year."),
        ("2. Finance and Lending",
         "The total operating budget for 2025 was 3.2 million euros. Staff costs accounted for 62 percent of the "
         "budget, collections for 21 percent, building costs for 9 percent and programmes for 8 percent. The "
         "library received 145,000 euros in donations. Readers borrowed about 1.1 million physical items. Digital "
         "lending of e-books and audiobooks reached 238,000 loans during the year."),
        ("3. Opening Hours, Events and Outlook",
         "The library is open Monday to Friday from 9:00 to 20:00 and on Saturday from 10:00 to 16:00. It is "
         "closed on Sundays. Late fees were abolished in June 2025. The Summer Reading Challenge attracted 2,140 "
         "participants. A new makerspace with 3 3D printers opened in autumn. The Eastside Annex branch is "
         "scheduled to open in March 2026."),
    ],
}


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    for filename, pages in DOCUMENTS.items():
        title = filename.replace(".pdf", "").replace("_", " ")
        doc = SimpleDocTemplate(str(OUT / filename), pagesize=A4, title=title, author="IntelliDocs sample data")
        story = []
        for i, (heading, body) in enumerate(pages):
            story += [Paragraph(heading, styles["Heading1"]), Spacer(1, 12), Paragraph(body, styles["BodyText"]),
                      Spacer(1, 24), Paragraph("<i>Fictional sample document for IntelliDocs testing.</i>", styles["Italic"])]
            if i < len(pages) - 1:
                story.append(PageBreak())
        doc.build(story)
        print("wrote", filename)


if __name__ == "__main__":
    build()
