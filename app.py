from flask import Flask, render_template, request, redirect, url_for, flash
import asyncio
from utils.feedback_utils import CompanyData, ResponseSystem

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Initialize the system (will be set in the index route)
system = None

@app.route("/", methods=["GET", "POST"])
def index():
    global system

    if request.method == "POST":
        company_name = request.form.get("company_name")
        faq_method = request.form.get("faq_method")
        faq_content = request.form.get("faq_content")
        faq_path = request.form.get("faq_path")
        product_categories = request.form.get("product_categories")

        if not company_name or (not faq_content and not faq_path) or not product_categories:
            flash("Please fill in all required fields.", "error")
            return redirect(url_for("index"))

        try:
            company_data = CompanyData(
                name=company_name,
                faq_content=faq_content if faq_method == "direct" else None,
                faq_path=faq_path if faq_method == "file" else None,
                product_categories=[cat.strip() for cat in product_categories.split(",")]
            )
            system = ResponseSystem(company_data)
            flash("System initialized successfully!", "success")
            return redirect(url_for("feedback"))
        except Exception as e:
            flash(f"Error initializing system: {str(e)}", "error")
            return redirect(url_for("index"))

    return render_template("index.html")

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    global system

    if not system:
        flash("Please initialize the system first.", "error")
        return redirect(url_for("index"))

    if request.method == "POST":
        feedback_text = request.form.get("feedback_text")
        if not feedback_text:
            flash("Please enter some feedback text.", "error")
            return redirect(url_for("feedback"))

        try:
            result = asyncio.run(system.process_feedback(feedback_text))
            return render_template("results.html", result=result)
        except Exception as e:
            flash(f"Error processing feedback: {str(e)}", "error")
            return redirect(url_for("feedback"))

    return render_template("feedback.html")

if __name__ == "__main__":
    app.run(debug=True)