import os
import io
import json
import logging
import sys
import traceback
import base64

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile

from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListItem, ListFlowable

import joblib

from .models import (
    SkinCondition, SkinCondition_page, User, PersonalizedPlan, Article
)
from users.models import DoctorReport, SkinProgress, MyAIReport
from utils.aliases import CONDITION_ALIASES

# --- END EXISTING IMPORTS ---

import base64
import json
import io
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.utils import timezone
from PIL import Image

from users.models import MyAIReport, SkinProgress
from .models import SkinCondition
from utils.aliases import CONDITION_ALIASES
import logging

logger = logging.getLogger(__name__)

# -----------------------------
# Hugging Face Space API
# -----------------------------
HF_SPACE_API = "https://huggingface.co/spaces/aleenaazam/skin-issues-prediction-model/run/predict"
# 1️⃣ Helper Functions (place them at the top)
def save_prediction_result(user, detected_issues, confidence_scores, image=None):
    avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0
    return SkinProgress.objects.create(
        user=user,
        image=image,
        detection_result=", ".join(detected_issues),
        ai_confidence=avg_confidence,
        confidence_scores_json=confidence_scores,
        improvement_score=0,
        age=getattr(user, 'age', 0),
        gender=getattr(user, 'gender', 'Not specified')
    )

def get_baseline_progress(user):
    try:
        return SkinProgress.objects.filter(user=user).order_by('-created_at').first()
    except Exception as e:
        logger.error(f"Error retrieving baseline progress: {e}")
        return None

def analyze_progress(current_confidence_scores, baseline_progress):
    if not baseline_progress or not hasattr(baseline_progress, 'confidence_scores_json') or not baseline_progress.confidence_scores_json:
        return {"status": "New analysis complete, no baseline data for comparison."}
    try:
        baseline_scores = json.loads(baseline_progress.confidence_scores_json)
    except Exception:
        return {"status": "New analysis complete, baseline data is corrupted."}
    improvements = {}
    regressions = {}
    for issue, current_score in current_confidence_scores.items():
        if issue in baseline_scores:
            baseline_score = baseline_scores[issue]
            score_change = current_score - baseline_score
            if score_change < -5.0:
                improvements[issue] = f"Reduced from {baseline_score:.1f}% to {current_score:.1f}%"
            elif score_change > 5.0:
                regressions[issue] = f"Increased from {baseline_score:.1f}% to {current_score:.1f}%"
    return {"status": "Comparison complete.", "improvements": improvements, "regressions": regressions}

@login_required
@csrf_exempt
@require_POST
def predict_view(request):
    """Handle image prediction via Hugging Face API"""
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)

    image_file = request.FILES['file']

    # Convert uploaded image to base64
    try:
        img = Image.open(image_file).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return JsonResponse({'error': 'Invalid image file'}, status=400)

    # Send to Hugging Face Space
    payload = {"data": [img_b64]}

    import requests
    try:
        response = requests.post(HF_SPACE_API, json=payload, timeout=60)
        response.raise_for_status()
        hf_result = response.json()
    except requests.RequestException as e:
        return JsonResponse({'error': f"Hugging Face API call failed: {e}"}, status=500)

    # Extract HF API results
    annotated_img_b64 = hf_result.get("annotated_image", "")
    detected_issues = [i.lower() for i in hf_result.get("detected_issues", [])]
    confidence_scores = hf_result.get("confidence_scores", {})

    # Expand misclassified groups
    all_issues = set(detected_issues)
    MISCLASS_GROUPS = {
        "dark circles": ["eye bags"], "eye bags": ["dark circles"],
        "psoriasis": ["eczema"], "eczema": ["psoriasis"],
        "sun spots": ["pigmentation"], "pigmentation": ["sun spots"],
        "acne": ["chicken pox"], "rosacea": ["acne"],
        "chicken pox": ["acne"], "warts": ["skin cancer"],
        "skin cancer": ["warts"], "shingles": ["skin cancer"]
    }
    for issue in detected_issues:
        if issue in MISCLASS_GROUPS:
            all_issues.update(MISCLASS_GROUPS[issue])

    # Save annotated image to Django model
    img_bytes = base64.b64decode(annotated_img_b64)
    annotated_file = ContentFile(img_bytes, name=f"annotated_{timezone.now().strftime('%Y%m%d%H%M%S')}.jpg")

    # Save SkinProgress & MyAIReport
    if request.user.is_authenticated:
        skin_progress = save_prediction_result(request.user, list(all_issues), confidence_scores, image=image_file)

        MyAIReport.objects.create(
            user=request.user,
            doctor=None,
            detected_issues=list(all_issues),
            confidence_scores=int(sum(confidence_scores.values())/len(confidence_scores) if confidence_scores else 0),
            confidence_scores_json=confidence_scores,
            annotated_image=annotated_file,
            skin_progress=skin_progress,
            prediction=", ".join(list(all_issues)),
            age=getattr(request.user, 'age', None),
            gender=getattr(request.user, 'gender', None)
        )

        # Compare with baseline
        baseline = get_baseline_progress(request.user)
        progress_summary = analyze_progress(confidence_scores, baseline) if baseline else {"status": "Analysis complete."}

    else:
        progress_summary = {"status": "User not authenticated."}

    # Fetch remedies
    remedies_data = {}
    for issue in all_issues:
        normalized_issue = CONDITION_ALIASES.get(issue.lower(), issue.lower())
        condition = SkinCondition.objects.filter(name__iexact=normalized_issue).first()
        if not condition:
            continue

        home_remedies = [
            {
                'title': r.title,
                'directions': r.directions or "",
                'amount': r.amount or "",
                'image_url': r.image.url if r.image else None
            } for r in condition.remedy_set.all()
        ]
        medical_remedies = [
            {
                'title': t.title,
                'directions': t.directions or "",
                'amount': t.amount or "",
                'image_url': t.image.url if t.image else None,
                'scientific_evidence': t.scientific_evidence
            } for t in condition.treatment_set.filter(category='medical')
        ]

        remedies_data[normalized_issue] = {
            'home_remedies': home_remedies,
            'medical_remedies': medical_remedies,
            'causes': [c.strip() for c in getattr(condition, 'causes', '').split('\n') if c.strip()],
            'symptoms': [s.strip() for s in getattr(condition, 'symptoms', '').split('\n') if s.strip()]
        }

    return JsonResponse({
        'status': 'success',
        'annotated_image': annotated_img_b64,
        'detected_issues': list(all_issues),
        'confidence_scores': confidence_scores,
        'remedies_data': remedies_data,
        'follow_up_questions': [],  # HF Space already runs YOLO + LR
        'progress_summary': progress_summary
    })


from django.http import HttpResponse
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListItem, ListFlowable
from .models import PersonalizedPlan

# Map your URL slugs to your Database condition_names
CONDITION_NAME_MAP = {
    "acne": "Acne",
    "rosacea": "Rosacea",
    "dark-circles": "Dark Circles",
    "pigmentation": "Pigmentation (Hyperpigmentation)",
    "wrinkles": "Wrinkles & Fine Lines",
    "blackheads": "Blackheads (Open Comedones)",
    "sun-spots": "Sun Spots (Solar Lentigines)",
    "eye-bags": "Eye Bags & Puffiness",
    "freckles": "Freckles (Ephelides)",
    "skin-cancer": "Skin Cancer Prevention & Post-Diagnosis Care",
    "psoriasis": "Psoriasis",
    "eczema": "Eczema (Atopic Dermatitis)",
    "shingles": "Shingles (Herpes Zoster)",
    "warts": "Warts (Common & Plantar)",
    "hives": "Hives (Urticaria)",
    "chicken-pox": "Chicken Pox (Varicella)"
} 

# Define a Professional Green Color Palette
DARK_GREEN = colors.HexColor("#376740")  # For Main Title
MEDIUM_GREEN = colors.HexColor("#209203") # For Condition Name
LIGHT_GREEN = colors.HexColor("#50aa1f")  # For Accent lines and Bullets

def download_lifestyle(request):
    conditions_param = request.GET.get('conditions', '')
    slugs = conditions_param.split(',') if conditions_param else []
    
    # Map slugs to DB names
    db_names = [CONDITION_NAME_MAP.get(s) for s in slugs if CONDITION_NAME_MAP.get(s)]
    
    # Fetch data from DB
    plans = PersonalizedPlan.objects.filter(condition_name__in=db_names)

    buffer = BytesIO()
    # SimpleDocTemplate handles page margins and flow automatically
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    story = []

    # --- CUSTOM STYLES ---
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=DARK_GREEN,
        spaceAfter=12,
        fontName="Helvetica-Bold"
    )
    
    condition_style = ParagraphStyle(
        'CondHeader',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=MEDIUM_GREEN,
        spaceBefore=15,
        spaceAfter=10,
        fontName="Helvetica-Bold"
    )

    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=DARK_GREEN,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14, # Line spacing
        textColor=colors.black
    )

    # --- BUILDING THE PDF CONTENT ---
    # Main Header
    story.append(Paragraph("Personalized Lifestyle Plan", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=LIGHT_GREEN, spaceAfter=20))

    if not plans.exists():
        story.append(Paragraph("No lifestyle plans found for selected conditions.", body_style))
    else:
        for plan in plans:
            # Condition Title
            story.append(Paragraph(f"Condition: {plan.condition_name}", condition_style))

            # Define the sections based on your model fields
            sections = [
                ("Dietary Advice", plan.diet),
                ("Skincare Routine", plan.skincare),
                ("Exercise Tips", plan.exercise),
                ("Sleep & Habits", plan.sleep)
            ]

            for title, content in sections:
                if content:
                    # Section Heading
                    story.append(Paragraph(title, section_style))
                    
                    # Process lines into a bulleted list
                    lines = str(content).split('\n')
                    bullet_items = []
                    for line in lines:
                        if line.strip():
                            # This creates the "-" bullet look with green color
                            bullet_items.append(
                                ListItem(
                                    Paragraph(line.strip(), body_style),
                                    bulletColor=LIGHT_GREEN,
                                    value="-"
                                )
                            )
                    
                    if bullet_items:
                        story.append(ListFlowable(bullet_items, bulletType='bullet', leftIndent=20))
                    
                    story.append(Spacer(1, 10))

            # Add space or Page Break between different conditions
            story.append(Spacer(1, 30))

    # Build PDF
    doc.build(story)
    
    pdf = buffer.getvalue()
    buffer.close()
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="lifestyle_plan.pdf"'
    response.write(pdf)
    return response


def lifestyle_plan_view(request, condition_name):
    """
    Returns the personalized lifestyle plan for a detected condition.
    """
    # Ensure proper capitalization / formatting if needed
    condition_name = condition_name.lower()  # 'acne', 'rosacea', etc.

    # Fetch the plan from DB
    plan = get_object_or_404(PersonalizedPlan, condition_name__iexact=condition_name)

    context = {
        'plan': plan
    }
    return render(request, 'lifestyle_pdf.html', context)



def capture(request):
    # This view function captures a single image from a webcam.
    cap = cv2.VideoCapture(0) # Initializes the video capture from the default camera (index 0).

    if not cap.isOpened():
        return HttpResponse("Webcam not accessible", status=500) # Returns an error if the camera can't be opened.

    ret, frame = cap.read() # Reads a single frame from the camera.
    cap.release() # Releases the camera resource.

    if not ret:
        return HttpResponse("Failed to capture image", status=500) # Returns an error if the capture failed.

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converts the frame from BGR (OpenCV default) to RGB.
    img_pil = Image.fromarray(frame_rgb) # Converts the NumPy array to a PIL Image.

    img_io = io.BytesIO() # Creates an in-memory byte stream.
    img_pil.save(img_io, format='JPEG') # Saves the image to the stream.
    img_io.seek(0) # Resets the stream pointer.

    return HttpResponse(img_io, content_type="image/jpeg") # Returns the image as an HTTP response.

def get_remedies(request):
    """
    Fetches and returns remedies (home + medical) and details for a specific skin condition.
    """
    issue = request.GET.get('issue', '').lower()  # Get the 'issue' parameter from URL

    try:
        condition = SkinCondition.objects.get(name__iexact=issue)

        # Causes and symptoms
        causes = [c.strip() for c in (condition.causes.split('\n') if condition.causes else []) if c.strip()]
        symptoms = [s.strip() for s in (condition.symptoms.split('\n') if condition.symptoms else []) if s.strip()]

        # Home remedies (from SkinCondition)
        home_remedies = [
            {
                "title": r.title,
                "directions": getattr(r, 'formatted_directions', lambda: r.directions)(),
                "amount": r.amount or "",
                "image_url": r.image.url if r.image else None
            }
            for r in condition.remedy_set.all()
        ]

        # Medical treatments (from Treatment model)
        medical_remedies = [
            {
                "title": t.title,
                "directions": t.directions or "",
                "amount": t.amount or "",
                "image_url": t.image.url if t.image else None,
                "scientific_evidence": t.scientific_evidence
            }
            for t in condition.treatment_set.filter(category='medical')
        ]

        # Debug log
        print(f"Sending remedies for '{issue}': home={len(home_remedies)}, medical={len(medical_remedies)}")

        return JsonResponse({
            "causes": causes,
            "symptoms": symptoms,
            "home_remedies": home_remedies,
            "medical_remedies": medical_remedies
        })

    except SkinCondition.DoesNotExist:
        return JsonResponse({'error': 'Condition not found'}, status=404)

# Your other view functions (e.g., user registration, login, etc.)
# ...

@login_required
def predict_page_view(request):
    # This view renders the main prediction page template.
    return render(request, 'predict.html')


from utils.aliases import CONDITION_ALIASES; # Re-importing a module; this line might be redundant if it's already at the top.

def article_general_view(request):
    # This view renders a generic article page template.
    return render(request, 'article.html', {}) # Renders the template with an empty context.

def skin_conditions_list_view(request):
    # This view fetches a list of all skin conditions from the database and renders a list page.
    all_skin_conditions = SkinCondition_page.objects.all() # Fetches all objects of the 'SkinCondition_page' model.

    context = {
        'all_skin_conditions': all_skin_conditions,
    }
    return render(request, 'Skin Conditions.html', context) # Renders the template with the list of conditions.


def skin_condition_detail(request, condition_slug):
    # This view fetches and displays a single skin condition's details based on its slug.
    skin_condition = get_object_or_404(SkinCondition_page, slug=condition_slug) # Fetches the object or returns a 404 error if not found.

    context = {
        'skin_condition': skin_condition,
    }
    return render(request, 'Skin Conditions.html', context) # Renders the detail page template.

from .models import Article # Imports the 'Article' model.

def article_detail(request, slug):
    # This view fetches a specific article by its slug and displays it.
    article = get_object_or_404(Article, slug=slug) # Fetches the article or returns a 404.

    # --- ADD THESE DEBUGGING LINES ---
    print(f"DEBUG: Type of 'article': {type(article)}")
    print(f"DEBUG: Content of 'article': {article}")
    if hasattr(article, 'pk'):
        print(f"DEBUG: PK of 'article': {article.pk}")
    else:
        print("DEBUG: 'article' object has no 'pk' attribute.")
    # --- END DEBUGGING LINES ---
    # These are temporary debugging lines to check the fetched object's details.

    related_articles = Article.objects.filter(
        category=article.category
    ).exclude(pk=article.pk)[:3] # Fetches up to 3 other articles from the same category, excluding the current one.

    return render(request, 'article_detail.html', {
        'article': article,
        'related_articles': related_articles
    }) # Renders the article detail page with the main article and related articles.




























