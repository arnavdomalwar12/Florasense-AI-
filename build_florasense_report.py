from pathlib import Path
import csv
import json
import textwrap

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    Flowable,
)


ROOT = Path(r"C:\Users\arnav\OneDrive\Desktop\DL Project")
OUT = ROOT / "FloraSense_AI_Complete_Project_Report.pdf"
OUTPUTS = ROOT / "outputs"


def read_csv(name):
    with (OUTPUTS / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pct(value):
    return f"{float(value):.3f}%"


class ArchitectureDiagram(Flowable):
    def __init__(self, width=170 * mm, height=105 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw_box(self, canvas, x, y, w, h, label, fill):
        canvas.setFillColor(fill)
        canvas.setStrokeColor(colors.HexColor("#285943"))
        canvas.roundRect(x, y, w, h, 7, fill=1, stroke=1)
        canvas.setFillColor(colors.HexColor("#123322"))
        canvas.setFont("Helvetica-Bold", 8)
        lines = label.split("\n")
        for i, line in enumerate(lines):
            canvas.drawCentredString(x + w / 2, y + h / 2 + (len(lines) - 1 - i) * 5, line)

    def arrow(self, canvas, x1, y1, x2, y2):
        canvas.setStrokeColor(colors.HexColor("#4c7c60"))
        canvas.setLineWidth(1.2)
        canvas.line(x1, y1, x2, y2)
        canvas.setFillColor(colors.HexColor("#4c7c60"))
        canvas.line(x2, y2, x2 - 4, y2 + 2)
        canvas.line(x2, y2, x2 - 4, y2 - 2)

    def draw(self):
        c = self.canv
        c.setFillColor(colors.HexColor("#f7fbf8"))
        c.roundRect(0, 0, self.width, self.height, 8, fill=1, stroke=0)
        w, h = 31 * mm, 17 * mm
        x = 5 * mm
        y = self.height - 28 * mm
        self.draw_box(c, x, y, w, h, "Input leaf\nimage", colors.HexColor("#dcefe2"))
        self.draw_box(c, x + 38 * mm, y, w, h, "OpenCV\nRGB + resize", colors.HexColor("#dcefe2"))
        self.draw_box(c, x + 76 * mm, y, w, h, "Normalize +\nTensor", colors.HexColor("#dcefe2"))
        self.arrow(c, x + w, y + h / 2, x + 38 * mm, y + h / 2)
        self.arrow(c, x + 69 * mm, y + h / 2, x + 76 * mm, y + h / 2)
        self.arrow(c, x + 107 * mm, y + h / 2, x + 114 * mm, y + h / 2)
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(colors.HexColor("#183c27"))
        c.drawString(5 * mm, self.height - 43 * mm, "Separate task-specific classifiers")
        for i, label in enumerate(
            ["Kaggle species\n12 classes", "Campus species\n10 classes", "PlantVillage disease\n38 classes"]
        ):
            bx = 5 * mm + i * 55 * mm
            self.draw_box(c, bx, self.height - 71 * mm, 45 * mm, 18 * mm, label, colors.HexColor("#e9f4ec"))
            self.arrow(c, x + 107 * mm, y - 2 * mm, bx + 22.5 * mm, self.height - 53 * mm)
        self.draw_box(c, 38 * mm, 8 * mm, 94 * mm, 17 * mm, "Top-k output + confidence +\nuncertainty / care-profile report", colors.HexColor("#cce6d3"))
        for bx in [27.5 * mm, 82.5 * mm, 137.5 * mm]:
            self.arrow(c, bx, self.height - 71 * mm, 85 * mm, 25 * mm)


class MetricBarChart(Flowable):
    def __init__(self, rows, width=170 * mm, height=78 * mm):
        super().__init__()
        self.rows = rows
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        left, bottom = 42 * mm, 12 * mm
        chart_w, chart_h = self.width - left - 5 * mm, self.height - bottom - 7 * mm
        c.setStrokeColor(colors.HexColor("#9db6a5"))
        c.line(left, bottom, left, bottom + chart_h)
        c.line(left, bottom, left + chart_w, bottom)
        c.setFont("Helvetica", 7)
        for tick in [0, 25, 50, 75, 100]:
            y = bottom + chart_h * tick / 100
            c.setStrokeColor(colors.HexColor("#e1ebe4"))
            c.line(left, y, left + chart_w, y)
            c.setFillColor(colors.HexColor("#456050"))
            c.drawRightString(left - 2, y - 2, str(tick))
        bar_h = min(7 * mm, chart_h / max(len(self.rows), 1) - 2)
        for i, row in enumerate(self.rows):
            y = bottom + chart_h - (i + 1) * (bar_h + 3)
            value = float(row["accuracy_percent"])
            width = chart_w * value / 100
            c.setFillColor(colors.HexColor("#4e9a68" if "mobilenet" in row["architecture"] else "#3f6fa5"))
            c.roundRect(left, y, width, bar_h, 2, fill=1, stroke=0)
            label = f'{row["task"]} / {row["architecture"].replace("_", " ")}'
            c.setFillColor(colors.HexColor("#25382b"))
            c.setFont("Helvetica", 6.5)
            c.drawRightString(left - 3, y + 2, label[:34])
            c.drawString(left + width + 2, y + 2, f"{value:.3f}%")


class FooterCanvas:
    pass


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d7e3da"))
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#607365"))
    canvas.drawString(18 * mm, 8 * mm, "FloraSense AI | Deep Learning Project Report")
    canvas.drawRightString(192 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleGreen", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=27, leading=32, textColor=colors.HexColor("#195b35"), alignment=TA_CENTER,
    spaceAfter=12,
))
styles.add(ParagraphStyle(
    name="Subtitle", parent=styles["Normal"], fontName="Helvetica",
    fontSize=12, leading=17, textColor=colors.HexColor("#456050"), alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    name="H1Green", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=17, leading=21, textColor=colors.HexColor("#195b35"), spaceBefore=10, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="H2Green", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=12, leading=15, textColor=colors.HexColor("#2b7146"), spaceBefore=7, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="BodyReport", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.5, leading=14, textColor=colors.HexColor("#26372b"), spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="SmallReport", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8, leading=11, textColor=colors.HexColor("#445b4b"), spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique",
    fontSize=8, leading=10, textColor=colors.HexColor("#65776a"), alignment=TA_CENTER,
))


def P(text, style="BodyReport"):
    return Paragraph(text, styles[style])


def bullets(items):
    return [P(f"• {item}") for item in items]


def make_table(data, widths=None, font_size=7.5):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f7147")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7cdbd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f7f2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def image(path, width, height=None):
    im = Image.open(path)
    if height is None:
        height = width * im.height / im.width
    return RLImage(str(path), width=width, height=height)


def build_story():
    test = read_csv("final_test_metrics.csv")
    efficiency = read_csv("model_efficiency_metrics.csv")
    story = []

    story += [
        Spacer(1, 16 * mm),
        P("FloraSense AI", "TitleGreen"),
        P("Complete Deep Learning Project Report", "Subtitle"),
        Spacer(1, 7 * mm),
        P("Plant species identification and crop disease classification using transfer learning", "Subtitle"),
        Spacer(1, 12 * mm),
        image(ROOT / "Mango.jpg", 62 * mm, 62 * mm),
        Spacer(1, 5 * mm),
        P("Prepared from the project notebook, saved checkpoints, manifests, reports, and measured experiment outputs.", "Caption"),
        Spacer(1, 16 * mm),
        make_table([
            ["Project", "FloraSense AI"],
            ["Primary framework", "PyTorch + Torchvision"],
            ["Main model", "MobileNetV3-Large"],
            ["Baseline model", "ResNet18"],
            ["Input size", "224 × 224 RGB image"],
            ["Recorded runtime", "Python 3.13.1, CPU"],
        ], [52 * mm, 105 * mm], 9),
        PageBreak(),
    ]

    story += [
        P("Executive Summary", "H1Green"),
        P("FloraSense AI is a computer-vision system that analyzes leaf images. It contains three independent classifiers: a 10-class Campus plant model, a 12-class Kaggle species model, and a 38-class PlantVillage crop-condition model. The models use ImageNet-pretrained MobileNetV3-Large and ResNet18 networks, OpenCV preprocessing, cross-entropy loss, Adam optimization, validation macro-F1 checkpoint selection, and held-out test evaluation."),
        P("The saved experiments report test accuracy between 99.065% and 99.767%. MobileNetV3-Large is the recommended deployment model because its checkpoints are about 16 MB and it has lower CPU latency than ResNet18. ResNet18 provides slightly higher measured test accuracy in the current experiments, but it is about 43 MB and is slower in the recorded benchmarks."),
        P("<b>Important interpretation:</b> these results describe performance on the project’s test sets. Real-world field performance requires additional validation using images from different phones, backgrounds, locations, cultivars, and disease stages."),
        P("Project goals", "H2Green"),
        *bullets([
            "Identify supported plant species from leaf images.",
            "Predict candidate crop disease classes using PlantVillage labels.",
            "Provide top-k predictions, confidence scores, and uncertainty warnings.",
            "Compare accuracy and deployment efficiency of two CNN architectures.",
            "Preserve reproducibility through manifests, class mappings, checkpoints, and environment metadata.",
        ]),
        PageBreak(),
    ]

    story += [
        P("1. Problem Definition and Scope", "H1Green"),
        P("Manual plant identification requires botanical expertise, while disease recognition can be difficult when symptoms are subtle or visually similar. FloraSense AI frames the task as supervised multi-class image classification."),
        make_table([
            ["Input", "Output"],
            ["One leaf or plant image", "Top-k plant species predictions"],
            ["Leaf image from a supported crop", "Candidate disease or healthy condition"],
            ["Prediction score", "Confidence-aware status and warning"],
            ["Recognized species", "Verified care profile when available"],
        ], [70 * mm, 87 * mm], 9),
        P("The three taxonomies remain separate. A species label from the Kaggle dataset is not silently merged with a Campus label, and a PlantVillage disease prediction is not automatically treated as compatible with an arbitrary species prediction."),
        P("Project limitations", "H2Green"),
        *bullets([
            "The classifier only knows classes present in its training labels.",
            "A softmax score is not automatically a calibrated probability.",
            "A PlantVillage output is a candidate visual classification, not a confirmed diagnosis.",
            "A care profile is returned only when manually entered and source-grounded.",
        ]),
        P("Example input images", "H2Green"),
        Table([
            [image(ROOT / "Arjun.JPG", 62 * mm, 41 * mm), image(ROOT / "Mango.jpg", 55 * mm, 55 * mm)],
            [P("Arjun.JPG: project input image", "Caption"), P("Mango.jpg: project input image", "Caption")],
        ], colWidths=[82 * mm, 82 * mm], style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])),
        PageBreak(),
    ]

    story += [
        P("2. System Architecture", "H1Green"),
        P("The architecture is a modular inference pipeline. Image decoding and preprocessing are shared, while the classification heads remain independent for each label vocabulary."),
        ArchitectureDiagram(),
        P("Figure 1. FloraSense AI end-to-end architecture.", "Caption"),
        P("Architectural principle", "H2Green"),
        P("The system uses task-specific models rather than one universal classifier. This prevents incompatible class indices from being mixed and allows each dataset to retain its own label semantics."),
        P("Inference report", "H2Green"),
        P("The final report can contain a species result from the supported species models, a separate PlantVillage candidate, top-k alternatives, a threshold-based uncertainty status, and a care profile lookup."),
        PageBreak(),
    ]

    story += [
        P("3. Dataset and Data Engineering Pipeline", "H1Green"),
        make_table([
            ["Dataset/task", "Classes", "Use"],
            ["Campus species", "10", "Common campus/garden plant identification"],
            ["Kaggle species", "12", "Indian plant species identification"],
            ["PlantVillage disease", "38", "Crop disease and healthy-condition classification"],
        ], [55 * mm, 25 * mm, 97 * mm], 8.5),
        P("Dataset pipeline", "H2Green"),
        *bullets([
            "Inspect dataset roots and class directories.",
            "Find image files with supported extensions.",
            "Reject unreadable images using OpenCV.",
            "Create a manifest containing path, label, dataset, and split metadata.",
            "Preserve Campus TRAIN/VALID/TEST folders when available.",
            "Create stratified 70:15:15 train/validation/test splits for class-root datasets.",
            "Generate and save a deterministic class-to-index mapping.",
        ]),
        P("Data leakage consideration", "H2Green"),
        P("If multiple photographs show the same physical leaf or plant, an image-level random split may place related images in both training and test data. A production-quality collection should split by plant identity, leaf identity, capture session, or location."),
        P("Preprocessing", "H2Green"),
        P("OpenCV decodes BGR images, converts them to RGB, and resizes them to 224 × 224. Training uses random rotation, color jitter, and horizontal flips. Validation and test images use deterministic tensor conversion and ImageNet normalization."),
        PageBreak(),
    ]

    story += [
        P("4. Deep Learning Concepts", "H1Green"),
        P("Convolutional neural networks learn hierarchical visual features. Early layers learn edges and color boundaries; middle layers learn veins, textures, spots, and margins; later layers combine these patterns into species or disease evidence."),
        make_table([
            ["Concept", "How it is used"],
            ["Supervised learning", "Images are paired with known class labels."],
            ["Multi-class classification", "Each model selects one class from its own vocabulary."],
            ["Transfer learning", "ImageNet-pretrained visual features are adapted to plant classes."],
            ["Cross-entropy loss", "Penalizes low probability assigned to the correct class."],
            ["Adam optimizer", "Updates model parameters using adaptive learning rates."],
            ["Weight decay", "Regularizes parameters and helps reduce overfitting."],
            ["Data augmentation", "Improves robustness to orientation and lighting variation."],
            ["Early stopping", "Stops when validation macro-F1 no longer improves."],
            ["Macro metrics", "Gives every class equal importance in the summary."],
        ], [46 * mm, 131 * mm], 8),
        P("MobileNetV3-Large", "H2Green"),
        P("MobileNetV3-Large is the main model because it is designed for efficient inference. It uses lightweight convolutional building blocks, inverted residual structures, and channel-attention mechanisms. It is well suited to mobile, CPU, and edge deployment."),
        P("ResNet18", "H2Green"),
        P("ResNet18 is the comparison baseline. Its residual connections learn a transformation F(x) and add the original input x, helping optimization in deeper networks."),
        PageBreak(),
    ]

    story += [
        P("5. Training Workflow and Code Explanation", "H1Green"),
        P("The notebook sets random seeds, selects CPU or GPU, builds manifests, creates Dataset and DataLoader objects, loads a pretrained network, replaces its final layer, and trains using mini-batch gradient descent."),
        P("Core training sequence", "H2Green"),
        make_table([
            ["Stage", "Operation"],
            ["Forward pass", "logits = model(images)"],
            ["Loss", "loss = CrossEntropyLoss(logits, labels)"],
            ["Backpropagation", "loss.backward()"],
            ["Parameter update", "optimizer.step()"],
            ["Validation", "Evaluate without updating weights"],
            ["Checkpointing", "Save the state with best validation macro-F1"],
        ], [42 * mm, 135 * mm], 8.5),
        P("The saved checkpoint includes architecture, model weights, class names, class mapping, image size, normalization statistics, best validation macro-F1, and seed. Keeping these fields together prevents incorrect labels or preprocessing during later inference."),
        P("Hyperparameters recorded in the environment", "H2Green"),
        make_table([
            ["Parameter", "Value"],
            ["Image size", "224 × 224"],
            ["Batch size", "32"],
            ["Learning rate", "0.0001"],
            ["Weight decay", "0.0001"],
            ["Maximum epochs", "10 in the recorded notebook configuration"],
            ["Early stopping patience", "4 epochs"],
            ["Random seed", "42"],
        ], [70 * mm, 107 * mm], 8.5),
        PageBreak(),
    ]

    story += [
        P("6. Evaluation Metrics", "H1Green"),
        P("<b>Accuracy</b> is the fraction of all predictions that are correct. <b>Precision</b> asks how often a predicted class is correct. <b>Recall</b> asks how many actual examples of a class were found. <b>F1-score</b> is the harmonic mean of precision and recall."),
        P("Formulas", "H2Green"),
        make_table([
            ["Metric", "Formula", "Interpretation"],
            ["Accuracy", "(TP + TN) / all samples", "Overall correctness"],
            ["Precision", "TP / (TP + FP)", "Controls false positives"],
            ["Recall", "TP / (TP + FN)", "Controls missed positives"],
            ["F1-score", "2PR / (P + R)", "Balances precision and recall"],
        ], [35 * mm, 55 * mm, 87 * mm], 8.5),
        P("Macro averaging calculates each class metric independently and averages the classes equally. This is important when the number of images differs between classes. The notebook also saves per-class classification reports and confusion matrices."),
        P("Operational metrics", "H2Green"),
        P("Model size, batch-1 latency, and FPS are also measured. The recorded latency benchmark measures the forward pass on an already preprocessed 224 × 224 tensor, so it does not include image decoding or OpenCV preprocessing."),
        PageBreak(),
    ]

    story += [
        P("7. Measured Test Results", "H1Green"),
        P("The following values are taken from the saved final test metrics. Percentages are shown exactly to three decimal places."),
    ]
    result_table = [["Task", "Architecture", "Images", "Accuracy", "Macro P", "Macro R", "Macro F1", "Size"]]
    for row in test:
        result_table.append([
            row["task"].replace("_", " "),
            row["architecture"].replace("_", " "),
            row["test_images"],
            pct(row["accuracy_percent"]),
            pct(row["macro_precision_percent"]),
            pct(row["macro_recall_percent"]),
            pct(row["macro_f1_percent"]),
            f'{float(row["checkpoint_size_MB"]):.3f} MB',
        ])
    story += [
        make_table(result_table, [28 * mm, 32 * mm, 17 * mm, 21 * mm, 21 * mm, 21 * mm, 21 * mm, 20 * mm], 6.7),
        Spacer(1, 5 * mm),
        MetricBarChart(test),
        P("Figure 2. Held-out test accuracy by task and architecture.", "Caption"),
        P("Best recorded accuracy", "H2Green"),
        *bullets([
            "Kaggle species: ResNet18, 99.377% accuracy.",
            "PlantVillage disease: ResNet18, 99.767% accuracy.",
            "Campus species: ResNet18, 99.752% accuracy.",
        ]),
        P("The results are excellent on the saved test sets, but they should not be interpreted as a guarantee of the same performance on uncontrolled field images.", "SmallReport"),
        PageBreak(),
    ]

    story += [
        P("8. Efficiency and Deployment Comparison", "H1Green"),
    ]
    eff_table = [["Task", "Architecture", "Latency", "FPS", "Size"]]
    for row in efficiency:
        eff_table.append([
            row["task"].replace("_", " "),
            row["architecture"].replace("_", " "),
            f'{float(row["latency_ms_batch1"]):.3f} ms',
            f'{float(row["fps_batch1"]):.2f}',
            f'{float(row["checkpoint_size_MB"]):.3f} MB',
        ])
    story += [
        make_table(eff_table, [46 * mm, 40 * mm, 28 * mm, 25 * mm, 30 * mm], 7.3),
        P("MobileNetV3-Large is the recommended deployment model because it is approximately 16 MB and generally gives higher CPU FPS. ResNet18 is useful when the small accuracy advantage is more important than memory and latency."),
        P("Recommended deployment choice", "H2Green"),
        make_table([
            ["Requirement", "Recommended model"],
            ["Mobile or edge device", "MobileNetV3-Large"],
            ["Low memory and fast CPU inference", "MobileNetV3-Large"],
            ["Maximum current benchmark accuracy", "ResNet18"],
            ["Research baseline comparison", "ResNet18"],
        ], [75 * mm, 102 * mm], 8.5),
        PageBreak(),
    ]

    story += [
        P("9. Inference, Confidence, and Care Workflow", "H1Green"),
        P("At inference time, the image follows the same resize and normalization path used during evaluation. The model is put into evaluation mode, logits are converted with softmax, and the top-k classes are returned."),
        P("The notebook uses a practical confidence threshold of 0.40. If the highest score is lower than the threshold, the result is reported as uncertain and the user is advised to retake the image or seek expert confirmation."),
        P("Combined analysis must be interpreted carefully. A PlantVillage disease model should only be applied when the predicted species/crop is compatible with its crop label. Otherwise the disease result is merely an unsupported candidate."),
        P("Care profiles", "H2Green"),
        P("The optional knowledge base stores common name, scientific name, description, soil, sunlight, watering, temperature, common problems, management notes, source URL, and review date. Unmatched labels return no profile rather than fabricated advice."),
        P("Suggested production response", "H2Green"),
        make_table([
            ["Field", "Example"],
            ["Species", "Mango"],
            ["Confidence", "0.96"],
            ["Alternatives", "Guava, Neem"],
            ["Status", "supported_prediction"],
            ["Disease", "Only if crop compatibility is confirmed"],
            ["Warning", "AI-assisted result; confirm with an expert"],
        ], [50 * mm, 127 * mm], 8.5),
        PageBreak(),
    ]

    story += [
        P("10. Benefits and Applications", "H1Green"),
        *bullets([
            "Agriculture: preliminary crop disease screening and field inspection assistance.",
            "Campus and gardens: plant inventory, biodiversity mapping, and educational identification.",
            "Nurseries: plant recognition and customer support.",
            "Environmental monitoring: local species cataloging and urban biodiversity surveys.",
            "Mobile AI: MobileNetV3-Large is small enough for offline and edge deployment.",
            "Education and research: demonstrates the full deep learning lifecycle from data preparation to deployment evaluation.",
        ]),
        P("Limitations and risk controls", "H2Green"),
        *bullets([
            "Controlled datasets may not represent outdoor backgrounds and field conditions.",
            "Similar species and disease symptoms can be confused.",
            "Unsupported species should be reported as uncertain rather than forced into a known class.",
            "Disease predictions must not be treated as confirmed diagnoses.",
            "Care and chemical-treatment information must be verified against local agricultural guidance.",
        ]),
        P("Recommended improvements", "H2Green"),
        *bullets([
            "Add a Streamlit, Gradio, or FastAPI user interface.",
            "Add leaf detection or segmentation for complex backgrounds.",
            "Collect real-world images across phones, regions, lighting, and disease stages.",
            "Use group-based dataset splitting to reduce leakage risk.",
            "Add Grad-CAM explanations and confidence calibration.",
            "Evaluate TorchScript, ONNX, quantization, and mobile deployment.",
        ]),
        PageBreak(),
    ]

    story += [
        P("11. Conclusion", "H1Green"),
        P("FloraSense AI implements a complete deep learning pipeline: dataset inspection, cleaning, manifest generation, stratified splitting, OpenCV preprocessing, augmentation, transfer learning, model training, validation, checkpoint selection, held-out testing, metric reporting, efficiency benchmarking, and image inference."),
        P("The saved test results show very high performance for the current datasets. MobileNetV3-Large is the most practical model for deployment because it offers a strong accuracy-efficiency balance. ResNet18 remains a useful accuracy baseline."),
        P("The next stage should focus on external validation. A deployment-ready system should be evaluated on real field images, include calibrated uncertainty, apply crop-compatible disease logic, and present the output as decision support rather than an unquestionable diagnosis."),
        P("Project artifact index", "H2Green"),
        make_table([
            ["Artifact", "Purpose"],
            ["FloraSense_Final_Input_Based_DL.ipynb", "Training, evaluation, inference, and benchmarking workflow"],
            ["models/*.pth", "Saved best model checkpoints"],
            ["outputs/*_manifest.csv", "Dataset paths, labels, and split records"],
            ["outputs/*_class_mapping.json", "Class-index lookup for inference"],
            ["outputs/final_test_metrics.csv", "Held-out test metrics"],
            ["outputs/model_efficiency_metrics.csv", "Latency, FPS, and checkpoint sizes"],
            ["outputs/*_classification_report.txt", "Per-class precision, recall, and F1"],
        ], [76 * mm, 101 * mm], 8),
        Spacer(1, 6 * mm),
        P("Source files used for this report: project notebook, saved model metrics, classification reports, environment metadata, class mappings, and the Arjun.JPG and Mango.jpg project images.", "SmallReport"),
    ]
    return story


def main():
    frame = Frame(
        18 * mm,
        18 * mm,
        A4[0] - 36 * mm,
        A4[1] - 34 * mm,
        id="normal",
    )
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="FloraSense AI Complete Deep Learning Project Report",
        author="FloraSense AI Project",
    )
    doc.addPageTemplates([PageTemplate(id="Report", frames=[frame], onPage=footer)])
    doc.build(build_story())
    print(OUT)


if __name__ == "__main__":
    main()
